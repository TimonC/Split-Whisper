import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
import torch.nn.functional as F
from load_data_custom_cslu import load_data_custom_cslu
from torch.amp import GradScaler, autocast
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== Model definition =====
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.skip = nn.Identity()
        if in_c != out_c or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

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

    def _make_layer(self, in_c, out_c, stride=1):
        return ResidualBlock(in_c, out_c, stride=stride)

    def downsample_mask(self, mask, target_length):
        m = mask.unsqueeze(1).float()
        factor = m.size(-1) // target_length
        if factor == 0:
            factor = 1
        m_ds = F.avg_pool1d(m, kernel_size=factor, stride=factor, ceil_mode=True)
        return (m_ds > 0.5).squeeze(1)

    def forward(self, x, mask=None):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if mask is not None:
            T_out = x.size(-1)
            m = self.downsample_mask(mask, T_out)
            m = m.unsqueeze(1).unsqueeze(2)
            x = x * m

        x = self.shared_fc(x)
        outputs = {}
        if self.task in ('age', 'both'):
            outputs['age'] = self.age_head(x).squeeze(1)
        if self.task in ('gender', 'both'):
            outputs['gender'] = self.gender_head(x).squeeze(1)
        return outputs

# ===== Data loading =====
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
            tmp = ds[split].add_column('y_age',[y_age]* len(ds[split]))
            tmp = tmp.add_column('y_gender', [y_gender] * len(tmp))
            coll.append(tmp)
    train = concatenate_datasets(all_train)
    dev   = concatenate_datasets(all_dev)
    return train, dev

# ===== Collate function =====
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

# ===== Training loop with dynamic weighting =====
def train_loop(model, loader, optimizer, loss_fn_age, loss_fn_gender, device, task='both'):
    model.train()
    scaler = GradScaler()
    total_loss = 0.0

    # Only initialize if task == 'both'
    if task == 'both':
        running_age_loss = 0.0
        running_gender_loss = 0.0
        beta = 0.99  # smoothing factor

    for feats, masks, ya, yg in tqdm(loader, desc='Train', leave=False):
        feats, masks = feats.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device.type):
            outputs = model(feats, mask=masks)

            if task == 'age':
                ya_device = ya.to(device, non_blocking=True).float()
                logits_age = outputs['age']
                loss = loss_fn_age(logits_age, ya_device)

            elif task == 'gender':
                yg_device = yg.to(device, non_blocking=True).float()
                logits_gender = outputs['gender']
                loss = loss_fn_gender(logits_gender, yg_device)

            elif task == 'both':
                # Compute both losses
                ya_device = ya.to(device, non_blocking=True).float()
                logits_age = outputs['age']
                loss_age = loss_fn_age(logits_age, ya_device)

                yg_device = yg.to(device, non_blocking=True).float()
                logits_gender = outputs['gender']
                loss_gender = loss_fn_gender(logits_gender, yg_device)

                # Update running averages
                running_age_loss = beta * running_age_loss + (1 - beta) * loss_age.detach().item()
                running_gender_loss = beta * running_gender_loss + (1 - beta) * loss_gender.detach().item()

                # Dynamic weighting
                denom = running_age_loss + running_gender_loss
                if denom > 0.0:
                    w_age = running_gender_loss / denom
                    w_gender = running_age_loss / denom
                else:
                    w_age, w_gender = 0.5, 0.5

                # Combine
                loss = w_age * loss_age + w_gender * loss_gender

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * feats.size(0)

    return total_loss / len(loader.dataset)

# ===== Evaluation loop =====
@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    preds, labels = {k: [] for k in ['age','gender']}, {k: [] for k in ['age','gender']}

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

    for k in preds:
        if preds[k]:
            preds[k]  = np.concatenate(preds[k])
            labels[k] = np.concatenate(labels[k])

    return preds, labels

# ===== Metrics =====
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
        arr_labels = np.array(labels['age'])
        arr_preds  = np.array(preds['age'])
        if class_weights is not None:
            overall_acc = weighted_accuracy(arr_preds, arr_labels, class_weights)
        else:
            overall_acc = accuracy_score(arr_labels, arr_preds)
        results['weighted_acc_all'] = overall_acc
        # Per-class accuracy
        for age_val, age_name in [(0, 'younger'), (1, 'older')]:
            idx = np.where(arr_labels == age_val)[0]
            results[f"acc_{age_name}"] = accuracy_score(arr_labels[idx], arr_preds[idx])
        return overall_acc, results

    elif task == 'gender':
        arr_labels = np.array(labels['gender'])
        arr_preds  = np.array(preds['gender'])
        if class_weights is not None:
            overall_acc = weighted_accuracy(arr_preds, arr_labels, class_weights)
        else:
            overall_acc = accuracy_score(arr_labels, arr_preds)
        results['weighted_acc_all'] = overall_acc
        # Per-class accuracy
        for gen_val, gen_name in [(0, 'girl'), (1, 'boy')]:
            idx = np.where(arr_labels == gen_val)[0]
            results[f"acc_{gen_name}"] = accuracy_score(arr_labels[idx], arr_preds[idx])
        return overall_acc, results

    else:  # task == 'both'
        # age weighted-acc
        age_arr  = np.array(labels['age'])
        age_pred = np.array(preds['age'])
        if class_weights is not None:
            acc_age = weighted_accuracy(age_pred, age_arr, class_weights[0])
        else:
            acc_age = accuracy_score(age_arr, age_pred)
        results['weighted_acc_age'] = acc_age

        # gender weighted-acc
        gen_arr  = np.array(labels['gender'])
        gen_pred = np.array(preds['gender'])
        if class_weights is not None:
            acc_gen = weighted_accuracy(gen_pred, gen_arr, class_weights[1])
        else:
            acc_gen = accuracy_score(gen_arr, gen_pred)
        results['weighted_acc_gender'] = acc_gen

        overall_acc = (acc_age + acc_gen) / 2.0
        return overall_acc, results

def compute_class_weights_from_counts(counts, num_classes):
    total = sum(counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        weights[i] = counts.get(i, 0) / total
    return weights

# ===== Main training function =====
def train_age_gender_classifier(args):
    train_ds, dev_ds = combine_datasets(args.dataset_path)
    for d in (train_ds, dev_ds):
        d.set_format(type='torch', columns=['input_features', 'y_age', 'y_gender'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y_age = train_ds['y_age'].clone()
    y_gender = train_ds['y_gender'].clone()

    # Compute pos_weight for BCE losses
    age_pos_weight   = ((y_age == 0).sum()) / ((y_age == 1).sum())
    gender_pos_weight = ((y_gender == 0).sum()) / ((y_gender == 1).sum())
    age_pos_weight   = age_pos_weight.to(device)
    gender_pos_weight = gender_pos_weight.to(device)

    loss_fn_age    = nn.BCEWithLogitsLoss(pos_weight=age_pos_weight)
    loss_fn_gender = nn.BCEWithLogitsLoss(pos_weight=gender_pos_weight)

    # Compute class counts and weights for metrics
    age_counts   = Counter(y_age.tolist())
    gender_counts = Counter(y_gender.tolist())
    age_weights_metric   = compute_class_weights_from_counts(age_counts, 2)
    gender_weights_metric = compute_class_weights_from_counts(gender_counts, 2)

    # Prepare a sampler to balance on joint classes if using task='both'
    group_labels = y_age * 2 + y_gender
    group_counts = Counter(group_labels.tolist())
    total_group = sum(group_counts.values())
    sampler_class_weights = {cls: total_group / (4 * group_counts[cls]) for cls in range(4)}
    sample_weights = [sampler_class_weights[int(lbl)] for lbl in group_labels.tolist()]
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
    best_results = {}
    losses = []

    for epoch in trange(args.num_train_epochs, desc='Epochs'):
        loss = train_loop(model, train_loader, optimizer, loss_fn_age, loss_fn_gender, device, task=args.task)
        losses.append(loss)

        preds, labels = eval_loop(model, dev_loader, device)

        # Choose class weights depending on the task
        if args.task == 'age':
            cw = age_weights_metric
        elif args.task == 'gender':
            cw = gender_weights_metric
        else:  # both
            cw = (age_weights_metric, gender_weights_metric)

        overall_acc, results = custom_metrics(preds, labels, args.task, cw)

        print(f"Epoch {epoch}: loss={loss:.4f} | weighted_acc={overall_acc:.4f}")

        if overall_acc > best_acc:
            best_acc = overall_acc
            patience_counter = 0
            best_results = results.copy()
            best_results["losses"] = losses.copy()
            if args.save_model:
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.task}.pt"))
        elif args.early_stopping:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    with open(json_file, 'w') as jf:
        json.dump(best_results, jf, indent=2)

    return best_results

# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',    type=str,   default='data_cslu_splits/gender/data/scripted')
    parser.add_argument('--output_dir',      type=str,   default='AgeGenderModels')
    parser.add_argument('--results_dir',     type=str,   default='results/classifier')
    parser.add_argument('--train_batch_size',type=int,   default=16)
    parser.add_argument('--eval_batch_size', type=int,   default=32)
    parser.add_argument('--num_train_epochs',type=int,   default=50)
    parser.add_argument('--learning_rate',   type=float, default=1e-4)
    parser.add_argument('--task',            type=str,   choices=['age','gender','both'], default='both')
    parser.add_argument('--patience',        type=int,   default=10, help='Early stopping patience')
    parser.add_argument('--save_model',      action='store_true', default=False)
    parser.add_argument('--seed',            type=int,   default=0)
    parser.add_argument('--early_stopping', action='store_true', default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Training a model to classify cslu children's speech by {args.task}. Seed: {args.seed}")
    train_age_gender_classifier(args)

if __name__ == '__main__':
    main()
