import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm, trange
import torch.nn.functional as F

from load_data_custom_cslu import load_data_custom_cslu

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
    def __init__(self):
        super().__init__()
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
        return self.age_head(x).squeeze(1), self.gender_head(x).squeeze(1)
# ===== Data Loading =====
def combine_datasets(dataset_path):
    class_names = ['younger_Boy', 'younger_Girl', 'older_Boy', 'older_Girl']

    all_train, all_dev = [], []
    for cls in class_names:
        path = os.path.join(dataset_path, cls)
        ds = load_data_custom_cslu(path, mode="train")

        is_younger = 'younger' in cls.lower()
        is_girl = 'girl' in cls.lower()
        y_age = int(not is_younger)   # 1 = older, 0 = younger
        y_gender = int(not is_girl)       # 1 = boy, 0 = girl

        for split, coll in [('train', all_train), ('development', all_dev)]:
            # Reset indices and avoid caching here:
            ds_split = ds[split].map(lambda x: x, load_from_cache_file=False)

            # Now add columns without triggering flattening:
            tmp = ds_split.add_column('y_age', [y_age] * len(ds_split))
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
        # label may not exist but both present in train
        ys_age.append(itm.get('y_age', None))
        ys_gen.append(itm.get('y_gender', None))
        pv = x.min().item()
        masks.append(~(torch.all(x == pv, dim=0)))
    batch_feats = torch.stack(feats).unsqueeze(1)
    batch_masks = torch.stack(masks)
    # convert labels
    ya = torch.tensor(ys_age) if ys_age[0] is not None else None
    yg = torch.tensor(ys_gen) if ys_gen[0] is not None else None
    return batch_feats, batch_masks, ya, yg


# ===== Training & Evaluation =====

def train_loop(model, ldr, opt, loss_fn, dev, task):
    model.train()
    total_loss = 0.0
    for feats, masks, ya, yg in tqdm(ldr, desc='Train', leave=False):
        feats = feats.to(dev)
        opt.zero_grad()
        outs = model(feats)
        loss = 0.0
        if task in ('age','both'):
            ya = ya.to(dev).float()
            loss += loss_fn(outs['age'], ya)
        if task in ('gender','both'):
            yg = yg.to(dev).float()
            loss += loss_fn(outs['gender'], yg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(ldr.dataset)

@torch.no_grad()
def eval_loop(model, ldr, dev, task):
    model.eval()
    preds, labels = {k: [] for k in ['age','gender','joint']}, {k: [] for k in ['age','gender','joint']}
    for feats, masks, ya, yg in tqdm(ldr, desc='Eval', leave=False):
        feats = feats.to(dev)
        outs = model(feats)
        if task in ('age','both'):
            pa = torch.sigmoid(outs['age'])
            preds['age'].append((pa>0.5).cpu().numpy())
            labels['age'].append(ya.numpy())
        if task in ('gender','both'):
            pg = torch.sigmoid(outs['gender'])
            preds['gender'].append((pg>0.5).cpu().numpy())
            labels['gender'].append(yg.numpy())
        if task=='both':
            pa = torch.sigmoid(outs['age']); pg = torch.sigmoid(outs['gender'])
            scores = torch.stack([(1-pa)*(1-pg),(1-pa)*pg,pa*(1-pg),pa*pg],dim=1)
            jp = scores.argmax(dim=1).cpu().numpy()
            jl = (ya*2 + yg).numpy()
            preds['joint'].append(jp)
            labels['joint'].append(jl)
    # concatenate
    for k in preds:
        if preds[k]: preds[k] = np.concatenate(preds[k]); labels[k] = np.concatenate(labels[k])
    return preds, labels


# custom evaluation metrics
def custom_metrics(preds, labels, task, losses=None):
    results = {}
    if task == 'age':
        classnames = ['younger', 'older']
    elif task == 'gender':
        classnames = ['girl', 'boy']
    elif task == 'both':
        classnames = ['younger_girl', 'younger_boy', 'older_girl', 'older_boy']
    else:
        raise ValueError(f"Unknown task: {task}")

    task_key = task if task != 'both' else 'joint'
    labels_arr = np.array(labels[task_key])
    preds_arr = np.array(preds[task_key])

    avg = 'weighted' if task == 'both' else 'binary'
    results['acc_all'] = accuracy_score(labels_arr, preds_arr)
    results['f1_all'] = f1_score(labels_arr, preds_arr, average=avg)

    for cls, cls_name in enumerate(classnames):
        idx = np.where(labels_arr == cls)[0]
        if len(idx) == 0:
            results[f"acc_{cls_name}"] = None
            results[f"f1_{cls_name}"] = None
            continue
        results[f"acc_{cls_name}"] = accuracy_score(labels_arr[idx], preds_arr[idx])
        results[f"f1_{cls_name}"] = f1_score(labels_arr[idx], preds_arr[idx], average=avg)

    results['loss'] = losses
    return results['acc_all'], results


# ===== Main Trainer =====
def train_age_gender_classifier(args):
    # Determine class names based on task
    train_ds, dev_ds = combine_datasets(args.dataset_path)
    for d in [train_ds, dev_ds]:
        d.set_format(type='torch', columns=['input_features', 'y_age', 'y_gender'])
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=hf_collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=hf_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinaryCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    # Prepare results structure
    max_epochs = args.num_train_epochs
    os.makedirs(args.results_dir, exist_ok=True)
    json_file = os.path.join(args.results_dir, f"{args.task}.json")

    best_acc = 0.0
    patience_counter = 0
    losses = []
    for epoch in trange(max_epochs, desc='Epochs'):
        loss = train_loop(model, train_loader, optimizer, loss_fn, device, args.task)
        losses.append(loss)
        preds, labels = eval_loop(model, dev_loader, device, args.task)
        # compute metrics
        overall_acc, results = custom_metrics(preds, labels, args.task, losses=losses)
        scheduler.step(overall_acc)

        # conditional model save
        if overall_acc > best_acc:
            patience_counter = 0
            best_acc = overall_acc
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.task}.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

        with open(json_file, 'w') as jf:
            json.dump(results, jf, indent=2)



# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data_cslu_splits/gender/data/scripted')
    parser.add_argument('--output_dir', type=str, default='AgeGenderModels')
    parser.add_argument('--results_dir', type=str, default='results/classifier')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--task', type=str, choices=['age','gender','both'], default='both')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_model', action='store_true', default=False)
    args = parser.parse_args()
    if args.task=='both':
        print(f"Training a model to classify clsu children's speech by age and gender.")
    else:
        print(f"Training a model to classify clsu children's speech by {args.task}.")
    train_age_gender_classifier(args)

if __name__ == '__main__':
    main()
