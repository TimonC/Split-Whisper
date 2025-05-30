import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from datasets import concatenate_datasets, Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm, trange
import torch.nn.functional as F
from load_data_custom_cslu import load_data_custom_cslu
from torch.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from collections import Counter
# ===== Model =====
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class BinaryCNN(nn.Module):
    def __init__(self, task='both'):
        super().__init__()
        self.task = task
        self.input_norm = nn.InstanceNorm2d(1, eps=1e-6, affine=False)
        self.initial = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(16, 32, stride=2)
        self.layer2 = self._make_layer(32, 64, stride=2)
        self.layer3 = self._make_layer(64, 128, stride=2)

        # shared pooling + fc
        self.shared_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.age_head    = nn.Linear(64, 1)
        self.gender_head = nn.Linear(64, 1)

    def _make_layer(self, in_c, out_c, stride=1):
        downsample = None
        if stride!=1 or in_c!=out_c:
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        return nn.Sequential(
            ResidualBlock(in_c, out_c, stride, downsample),
            ResidualBlock(out_c, out_c)
        )

    def downsample_mask(self, mask, target_length):
        m = mask.unsqueeze(1).float()                   # (N,1,T_in)
        factor = m.size(-1) // target_length
        m_ds = F.avg_pool1d(m, kernel_size=factor, stride=factor)
        return (m_ds > 0.5).squeeze(1)                  # (N, T_out)

    def forward(self, x, mask=None):
        x = self.input_norm(x)
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if mask is not None:
            T_out = x.size(-1)
            m = self.downsample_mask(mask, T_out)      # (N, T_out)
            m = m.unsqueeze(1).unsqueeze(2)            # (N,1,1,T_out)
            x = x * m                                  # broadcast over C,H

        x = self.shared_fc(x)                          # → (N,64)

        outputs = {}
        if self.task in ('age', 'both'):
            outputs['age'] = self.age_head(x).squeeze(1)
        if self.task in ('gender', 'both'):
            outputs['gender'] = self.gender_head(x).squeeze(1)
        return outputs

# ===== Data Loading =====
def combine_datasets(dataset_path):
    class_names = ['younger_Girl', 'younger_Boy', 'older_Girl', 'older_Boy']
    all_train, all_dev = [], []
    for cls in class_names:
        ds = load_data_custom_cslu(os.path.join(dataset_path, cls), mode='train')
        is_younger = 'younger' in cls.lower()
        is_girl = 'girl' in cls.lower()
        y_age = int(not is_younger)   # 1 = older, 0 = younger
        y_gender = int(not is_girl)       # 1 = boy, 0 = girl
        for split, coll in [('train', all_train), ('development', all_dev)]:
            tmp = ds[split].add_column('y_age', [y_age] * len(ds[split]))
            tmp = tmp.add_column('y_gender', [y_gender] * len(tmp))
            coll.append(tmp)
    train = concatenate_datasets(all_train)
    dev = concatenate_datasets(all_dev)
    return train, dev
# ===== Collate =====
def hf_collate_fn(batch):
    feats, ys_age, ys_gen, masks = [], [], [], []
    for itm in batch:
        x = itm['input_features']
        x = x.clone().float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        feats.append(x)
        ys_age.append(itm.get('y_age', None))
        ys_gen.append(itm.get('y_gender', None))
        pv = x.min().item()
        masks.append(~(torch.all(x == pv, dim=0)))
    batch_feats = torch.stack(feats).unsqueeze(1)
    batch_masks = torch.stack(masks)
    ya = torch.tensor(ys_age) if ys_age[0] is not None else None
    yg = torch.tensor(ys_gen) if ys_gen[0] is not None else None
    return batch_feats, batch_masks, ya, yg

# ===== Training & Evaluation =====

def train_loop(model, ldr, opt, loss_fn, dev):
    model.train()
    scaler = GradScaler()
    total_loss = 0.0
    for feats, masks, ya, yg in tqdm(ldr, desc='Train', leave=False):
        feats, masks = feats.to(dev, non_blocking=True), masks.to(dev, non_blocking=True)
        opt.zero_grad()
        with autocast(dev.type):
            outs = model(feats, mask=masks)
            loss = 0.0
            if model.task in ('age', 'both'):
                ya = ya.to(dev, non_blocking=True).float()
                loss += loss_fn(outs['age'], ya)
            if model.task in ('gender', 'both'):
                yg = yg.to(dev, non_blocking=True).float()
                loss += loss_fn(outs['gender'], yg)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(ldr.dataset)

@torch.no_grad()
def eval_loop(model, ldr, dev):
    model.eval()
    preds, labels = {k: [] for k in ['age','gender','joint']}, {k: [] for k in ['age','gender','joint']}
    for feats, masks, ya, yg in tqdm(ldr, desc='Eval', leave=False):
        feats, masks = feats.to(dev), masks.to(dev)
        outs = model(feats, mask=masks)
        if model.task in ('age','both'):
            pa = torch.sigmoid(outs['age'])
            preds['age'].append((pa>0.5).cpu().numpy())
            labels['age'].append(ya.numpy())
        if model.task in ('gender','both'):
            pg = torch.sigmoid(outs['gender'])
            preds['gender'].append((pg>0.5).cpu().numpy())
            labels['gender'].append(yg.numpy())
        if model.task=='both':
            pa = torch.sigmoid(outs['age']); pg = torch.sigmoid(outs['gender'])
            scores = torch.stack([(1-pa)*(1-pg),(1-pa)*pg,pa*(1-pg),pa*pg],dim=1)
            jp = scores.argmax(dim=1).cpu().numpy()
            jl = (ya*2 + yg).numpy()
            preds['joint'].append(jp)
            labels['joint'].append(jl)
    for k in preds:
        if preds[k]: preds[k] = np.concatenate(preds[k]); labels[k] = np.concatenate(labels[k])
    return preds, labels

# custom evaluation metrics
def custom_metrics(preds, labels, task):
    results = {}
    if task == 'age':
        classnames = ['younger', 'older']
    elif task == 'gender':
        classnames = ['girl', 'boy']
    else:
        classnames = ['younger_girl', 'younger_boy', 'older_girl', 'older_boy']

    key = 'joint' if task=='both' else task
    labels_arr = np.array(labels[key]); preds_arr = np.array(preds[key])
    avg = 'weighted' if task=='both' else 'binary'
    results['acc_all'] = accuracy_score(labels_arr, preds_arr)
    results['f1_all'] = f1_score(labels_arr, preds_arr, average=avg)
    for i, name in enumerate(classnames):
        idx = np.where(labels_arr==i)[0]
        if len(idx)==0:
            results[f"acc_{name}"]=None; results[f"f1_{name}"]=None
        else:
            results[f"acc_{name}"]=accuracy_score(labels_arr[idx], preds_arr[idx])
            results[f"f1_{name}"]=f1_score(labels_arr[idx], preds_arr[idx], average=avg)
    return results['acc_all'], results

# ===== Main Trainer =====
def train_age_gender_classifier(args):
    train_ds, dev_ds = combine_datasets(args.dataset_path)
    for d in (train_ds, dev_ds):
        d.set_format(type='torch', columns=['input_features', 'y_age', 'y_gender'])

       
    # --- Oversampling setup ---
    if args.task == 'age':
        labels = [int(l) for l in train_ds['y_age']]
    elif args.task == 'gender':
        labels = [int(l) for l in train_ds['y_gender']]
    else:  # both → combine age and gender into joint label: 0-3
        labels = [2 * int(a) + int(g) for a, g in zip(train_ds['y_age'], train_ds['y_gender'])]

    label_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    ) 
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, sampler=sampler,  collate_fn=hf_collate_fn, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=hf_collate_fn, num_workers=4, pin_memory=True)

    global dev
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinaryCNN(task=args.task).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(args.results_dir, exist_ok=True)
    json_file = os.path.join(args.results_dir, f"{args.task}.json")

    best_acc, patience_counter, losses = 0.0, 0, []
    best_results = {}
    for epoch in trange(args.num_train_epochs, desc='Epochs'):
        loss = train_loop(model, train_loader, optimizer, loss_fn, dev)
        losses.append(loss)
        preds, labels = eval_loop(model, dev_loader, dev)
        overall_acc, results = custom_metrics(preds, labels, args.task)
        scheduler.step(overall_acc)
        print(f" it{epoch} --- loss={loss} | overall_acc={overall_acc}")
        if overall_acc > best_acc:
            best_acc = overall_acc; patience_counter = 0; best_results = results
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.task}.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
    best_results["losses"] = losses
    with open(json_file, 'w') as jf:
        json.dump(best_results, jf, indent=2)

# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data_cslu_splits/gender/data/scripted')
    parser.add_argument('--output_dir', type=str, default='AgeGenderModels')
    parser.add_argument('--results_dir', type=str, default='results/classifier')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--task', type=str, choices=['age','gender','both'], default='both')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_model', action='store_true', default=False)
    args = parser.parse_args()
    print(f"Training a model to classify cslu children's speech by {args.task}.")
    train_age_gender_classifier(args)

if __name__ == '__main__':
    main()
