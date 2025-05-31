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
class BinaryCNN(nn.Module):
    def __init__(self, task='both'):
        super().__init__()
        self.task = task

        self.initial = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(8, 16, stride=2)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)

        self.shared_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.age_head    = nn.Linear(32, 1)
        self.gender_head = nn.Linear(32, 1)
        # Joint 4‐class head: younger_girl, younger_boy, older_girl, older_boy
        self.group_head  = nn.Linear(32, 4)

    def _make_layer(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def downsample_mask(self, mask, target_length):
        m = mask.unsqueeze(1).float()  # (N,1,T_in)
        factor = m.size(-1) // target_length
        if factor == 0:
            factor = 1
        m_ds = F.avg_pool1d(m, kernel_size=factor, stride=factor, ceil_mode=True)
        return (m_ds > 0.5).squeeze(1)  # (N, T_out)

    def forward(self, x, mask=None):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # (N, C, F, T_out)

        if mask is not None:
            T_out = x.size(-1)
            m = self.downsample_mask(mask, T_out)  # (N, T_out)
            m = m.unsqueeze(1).unsqueeze(2)        # (N,1,1,T_out)
            x = x * m                              # broadcast over C, F

        x = self.shared_fc(x)  # → (N, 32)

        outputs = {}
        if self.task in ('age', 'both'):
            outputs['age'] = self.age_head(x).squeeze(1)
        if self.task in ('gender', 'both'):
            outputs['gender'] = self.gender_head(x).squeeze(1)
        if self.task == 'both':
            outputs['group'] = self.group_head(x)  # Joint logits
        return outputs

# ===== Data Loading =====
def combine_datasets(dataset_path):
    class_names = ['younger_Girl', 'younger_Boy', 'older_Girl', 'older_Boy']
    all_train, all_dev = [], []
    for cls in class_names:
        ds = load_data_custom_cslu(os.path.join(dataset_path, cls), mode='train')
        is_younger = 'younger' in cls.lower()
        is_girl    = 'girl' in cls.lower()
        y_age      = int(not is_younger)   # 1 = older, 0 = younger
        y_gender   = int(not is_girl)      # 1 = boy, 0 = girl
        for split, coll in [('train', all_train), ('development', all_dev)]:
            tmp = ds[split].add_column('y_age',    [y_age]    * len(ds[split]))
            tmp = tmp.add_column('y_gender', [y_gender] * len(tmp))
            coll.append(tmp)
    train = concatenate_datasets(all_train)
    dev   = concatenate_datasets(all_dev)
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

# ===== Training & Evaluation Loops =====
def train_loop(model, loader, optimizer, loss_fn_age, loss_fn_gender, loss_fn_group, device):
    model.train()
    scaler = GradScaler()
    total_loss = 0.0

    for feats, masks, ya, yg in tqdm(loader, desc='Train', leave=False):
        feats, masks = feats.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device.type):
            outputs = model(feats, mask=masks)
            loss = 0.0

            if model.task in ('age', 'both'):
                ya = ya.to(device, non_blocking=True).float()
                loss += loss_fn_age(outputs['age'], ya)
            if model.task in ('gender', 'both'):
                yg = yg.to(device, non_blocking=True).float()
                loss += loss_fn_gender(outputs['gender'], yg)
            if model.task == 'both':
                group_labels = (ya.long() * 2 + yg.long()).to(device)  # 0..3
                loss += loss_fn_group(outputs['group'], group_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * feats.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    preds, labels = {k: [] for k in ['age','gender','joint']}, {k: [] for k in ['age','gender','joint']}

    for feats, masks, ya, yg in tqdm(loader, desc='Eval', leave=False):
        feats, masks = feats.to(device), masks.to(device)
        outputs = model(feats, mask=masks)

        if model.task in ('age','both'):
            pa = torch.sigmoid(outputs['age'])
            preds['age'].append((pa > 0.5).cpu().numpy())
            labels['age'].append(ya.numpy())

        if model.task in ('gender','both'):
            pg = torch.sigmoid(outputs['gender'])
            preds['gender'].append((pg > 0.5).cpu().numpy())
            labels['gender'].append(yg.numpy())

        if model.task == 'both':
            group_logits = outputs['group']
            jp = group_logits.argmax(dim=1).cpu().numpy()
            jl = (ya * 2 + yg).numpy()
            preds['joint'].append(jp)
            labels['joint'].append(jl)

    for k in preds:
        if preds[k]:
            preds[k]  = np.concatenate(preds[k])
            labels[k] = np.concatenate(labels[k])

    return preds, labels

# ===== Main Training Functi# ===== Metrics =====
def weighted_accuracy(preds, labels, class_weights):
    weighted_acc = 0.0
    total_weight = class_weights.sum()

    for cls_idx, weight in enumerate(class_weights):
        idx = np.where(labels == cls_idx)[0]
        if len(idx) == 0:
            acc = 0.0
        else:
            acc = accuracy_score(labels[idx], preds[idx])
        weighted_acc += acc * (weight / total_weight)

    return weighted_acc

def custom_metrics(preds, labels, task, class_weights=None):
    results = {}
    if task == 'age':
        classnames = ['younger', 'older']
    elif task == 'gender':
        classnames = ['girl', 'boy']
    else:
        classnames = ['younger_girl', 'younger_boy', 'older_girl', 'older_boy']

    key = 'joint' if task == 'both' else task
    labels_arr = np.array(labels[key])
    preds_arr  = np.array(preds[key])
    avg = 'macro' if task == 'both' else 'binary'

    if class_weights is not None:
        overall_acc = weighted_accuracy(preds_arr, labels_arr, class_weights)
    else:
        overall_acc = accuracy_score(labels_arr, preds_arr)

    results['weighted_acc_all'] = overall_acc
    results['f1_all'] = f1_score(labels_arr, preds_arr, average=avg)

    for i, name in enumerate(classnames):
        idx = np.where(labels_arr == i)[0]
        if len(idx) == 0:
            results[f"acc_{name}"] = None
            results[f"f1_{name}"] = None
        else:
            results[f"acc_{name}"] = accuracy_score(labels_arr[idx], preds_arr[idx])
            results[f"f1_{name}"] = f1_score(labels_arr[idx], preds_arr[idx], average=avg)

    return overall_acc, results

# ===== Main Training Function =====
def train_age_gender_classifier(args):
    train_ds, dev_ds = combine_datasets(args.dataset_path)
    for d in (train_ds, dev_ds):
        d.set_format(type='torch', columns=['input_features', 'y_age', 'y_gender'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute pos_weight for BCE losses (unchanged)
    y_age = train_ds['y_age'].clone()
    y_gender = train_ds['y_gender'].clone()
    age_pos_weight = ((y_age == 0).sum()) / ((y_age == 1).sum())
    gender_pos_weight = ((y_gender == 0).sum()) / ((y_gender == 1).sum())
    age_pos_weight = age_pos_weight.to(device)
    gender_pos_weight = gender_pos_weight.to(device)

    loss_fn_age = nn.BCEWithLogitsLoss(pos_weight=age_pos_weight)
    loss_fn_gender = nn.BCEWithLogitsLoss(pos_weight=gender_pos_weight)

    # Compute class weights for joint classes for loss and weighted accuracy
    group_labels_tensor = y_age * 2 + y_gender
    counts = Counter(group_labels_tensor.tolist())
    total = sum(counts.values())

    # For CrossEntropyLoss weights (inverse frequency)
    class_weights_loss = torch.tensor([total / counts[i] for i in range(4)], dtype=torch.float32).to(device)
    loss_fn_group = nn.CrossEntropyLoss(weight=class_weights_loss)

    # For weighted accuracy (normalized frequency)
    class_weights_metric = torch.tensor([counts[i] / total for i in range(4)], dtype=torch.float32)

    # WeightedRandomSampler setup (unchanged)
    num_classes = 4
    sampler_class_weights = {cls: total / (num_classes * counts[cls]) for cls in range(4)}
    sample_weights = [sampler_class_weights[int(lbl)] for lbl in group_labels_tensor.tolist()]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        sampler=sampler,
        collate_fn=hf_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=hf_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = BinaryCNN(task=args.task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    os.makedirs(args.results_dir, exist_ok=True)
    json_file = os.path.join(args.results_dir, f"{args.task}.json")

    best_acc, patience_counter = 0.0, 0
    losses, best_results = [], {}

    for epoch in trange(args.num_train_epochs, desc='Epochs'):
        loss = train_loop(model, train_loader, optimizer,
                          loss_fn_age, loss_fn_gender, loss_fn_group, device)
        losses.append(loss)

        preds, labels = eval_loop(model, dev_loader, device)
        overall_acc, results = custom_metrics(preds, labels, args.task, class_weights_metric)
        print(f"Epoch {epoch}: loss={loss:.4f} | weighted_acc={overall_acc:.4f}")

        if overall_acc > best_acc:
            best_acc = overall_acc
            patience_counter = 0
            best_results = results
            if args.save_model:
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.task}.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    best_results["losses"] = losses

    for k, v in best_results.items():
        if isinstance(v, torch.Tensor):
            best_results[k] = v.item()
        elif isinstance(v, list) and any(isinstance(x, torch.Tensor) for x in v):
            best_results[k] = [x.item() if isinstance(x, torch.Tensor) else x for x in v]
    with open(json_file, 'w') as jf:
        json.dump(best_results, jf, indent=2)
# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data_cslu_splits/gender/data/scripted')
    parser.add_argument('--output_dir',    type=str, default='AgeGenderModels')
    parser.add_argument('--results_dir',   type=str, default='results/classifier')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size',  type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--learning_rate',    type=float, default=1e-4)
    parser.add_argument('--task', type=str, choices=['age','gender','both'], default='both')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_model', action='store_true', default=False)
    args = parser.parse_args()
    print(f"Training a model to classify cslu children's speech by {args.task}.")
    train_age_gender_classifier(args)

if __name__ == '__main__':
    main()
