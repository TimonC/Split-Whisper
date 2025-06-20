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

# ===== CNN for binary classification with an age head and a gender head =====
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

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
        out = self.relu(out)
        return out


class BinaryResNet(nn.Module):
    def __init__(self, task='both'):
        super().__init__()
        self.task = task

        # Shared trunk: initial conv + 2 residual blocks
        self.initial = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(8, 16, stride=2)
        self.layer2 = ResBlock(16, 32, stride=2)

        # Separate third residual blocks for each head
        self.layer3_age = ResBlock(32, 64, stride=2)
        self.layer3_gender = ResBlock(32, 64, stride=2)

        # Age head: adaptive pooling → flatten → MLP → output
        self.age_head_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.age_head = nn.Linear(32, 1)

        # Gender head: same structure as age
        self.gender_head_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.gender_head = nn.Linear(32, 1)

    def _downsample_mask(self, mask, tgt_len):
        # mask: (B, T) → (B, tgt_len)
        m = mask.unsqueeze(1).float()                          # → (B, 1, T)
        factor = max(m.size(-1) // tgt_len, 1)
        m_ds = F.avg_pool1d(m, kernel_size=factor, stride=factor, ceil_mode=True)
        return (m_ds > 0.5).squeeze(1)                         # → (B, tgt_len)

    def forward(self, x, mask=None):
        # x: (B, 1, T, F), mask: (B, T) or None
        shared = self.initial(x)
        shared = self.layer1(shared)
        shared = self.layer2(shared)

        outputs = {}

        if self.task in ('age', 'both'):
            xa = self.layer3_age(shared)                         # (B, 64, T’, F’)
            if mask is not None:
                m_a = self._downsample_mask(mask, xa.size(-1))  # (B, T’)
                xa = xa * m_a.unsqueeze(1).unsqueeze(2)         # broadcast mask on channels and freq dim
            fa = self.age_head_fc(xa)                            # (B, 32)
            outputs['age'] = self.age_head(fa).squeeze(1)       # (B,)

        if self.task in ('gender', 'both'):
            xg = self.layer3_gender(shared)                      # (B, 64, T’, F’)
            if mask is not None:
                m_g = self._downsample_mask(mask, xg.size(-1))  # (B, T’)
                xg = xg * m_g.unsqueeze(1).unsqueeze(2)         # broadcast mask on channels and freq dim
            fg = self.gender_head_fc(xg)                         # (B, 32)
            outputs['gender'] = self.gender_head(fg).squeeze(1)# (B,)

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
                ya_device = ya.to(device, non_blocking=True).float()
                logits_age = outputs['age']
                loss_age = loss_fn_age(logits_age, ya_device)

                yg_device = yg.to(device, non_blocking=True).float()
                logits_gender = outputs['gender']
                loss_gender = loss_fn_gender(logits_gender, yg_device)

                running_age_loss = beta * running_age_loss + (1 - beta) * loss_age.detach().item()
                running_gender_loss = beta * running_gender_loss + (1 - beta) * loss_gender.detach().item()

                denom = running_age_loss + running_gender_loss
                if denom > 0.0:
                    w_age = running_gender_loss / denom
                    w_gender = running_age_loss / denom
                else:
                    w_age, w_gender = 0.5, 0.5

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

def custom_metrics(preds, labels, task, class_weights):
    results = {}

    if task == 'age':
        arr_labels = np.array(labels['age'])
        arr_preds  = np.array(preds['age'])
        overall_acc = weighted_accuracy(arr_preds, arr_labels, class_weights)
        results['weighted_acc_all'] = float(overall_acc)

        for age_val, age_name in [(0, 'younger'), (1, 'older')]:
            idx = np.where(arr_labels == age_val)[0]
            results[f"acc_{age_name}"] = float(accuracy_score(arr_labels[idx], arr_preds[idx]))
        return overall_acc, results

    elif task == 'gender':
        arr_labels = np.array(labels['gender'])
        arr_preds  = np.array(preds['gender'])
        overall_acc = weighted_accuracy(arr_preds, arr_labels, class_weights)
        results['weighted_acc_all'] = float(overall_acc)

        for gen_val, gen_name in [(0, 'girl'), (1, 'boy')]:
            idx = np.where(arr_labels == gen_val)[0]
            results[f"acc_{gen_name}"] = float(accuracy_score(arr_labels[idx], arr_preds[idx]))
        return overall_acc, results

    else:  # task == 'both'
        age_arr   = np.array(labels['age'])
        age_pred  = np.array(preds['age'])
        acc_age   = weighted_accuracy(age_pred, age_arr, class_weights[0])
        results['weighted_acc_age'] = float(acc_age)

        gen_arr   = np.array(labels['gender'])
        gen_pred  = np.array(preds['gender'])
        acc_gen   = weighted_accuracy(gen_pred, gen_arr, class_weights[1])
        results['weighted_acc_gender'] = float(acc_gen)

        # per-age accuracies
        for age_val, age_name in [(0, 'younger'), (1, 'older')]:
            idx = np.where(age_arr == age_val)[0]
            results[f"acc_{age_name}"] = float(accuracy_score(age_arr[idx], age_pred[idx]))

        # per-gender accuracies
        for gen_val, gen_name in [(0, 'girl'), (1, 'boy')]:
            idx = np.where(gen_arr == gen_val)[0]
            results[f"acc_{gen_name}"] = float(accuracy_score(gen_arr[idx], gen_pred[idx]))

        overall_acc = (acc_age + acc_gen) / 2.0
        results[f"weighted_acc_all"] = float(overall_acc);
        return overall_acc, results


def compute_class_weights_from_counts(counts, num_classes):
    total = sum(counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        weights[i] = counts.get(i, 0) / total
    return weights

# ===== Train loop =====
def train_age_gender_classifier(args):
    train_ds, dev_ds = combine_datasets(args.dataset_path)
    for d in (train_ds, dev_ds):
        d.set_format(type='torch', columns=['input_features', 'y_age', 'y_gender'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y_age = train_ds['y_age'].clone()
    y_gender = train_ds['y_gender'].clone()

    age_pos_weight   = ((y_age == 0).sum()) / ((y_age == 1).sum())
    gender_pos_weight = ((y_gender == 0).sum()) / ((y_gender == 1).sum())
    age_pos_weight   = age_pos_weight.to(device)
    gender_pos_weight = gender_pos_weight.to(device)

    loss_fn_age    = nn.BCEWithLogitsLoss(pos_weight=age_pos_weight)
    loss_fn_gender = nn.BCEWithLogitsLoss(pos_weight=gender_pos_weight)

    age_counts   = Counter(y_age.tolist())
    gender_counts = Counter(y_gender.tolist())
    age_weights_metric   = compute_class_weights_from_counts(age_counts, 2)
    gender_weights_metric = compute_class_weights_from_counts(gender_counts, 2)

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

    model = BinaryResNet(task=args.task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    os.makedirs(args.results_dir, exist_ok=True)
    json_file = os.path.join(args.results_dir, f"{args.task}.json")

    best_acc, patience_counter = 0.0, 0
    best_results = {}
    losses = []

    for epoch in trange(args.num_train_epochs, desc='Epochs'):
        loss = train_loop(model, train_loader, optimizer, loss_fn_age, loss_fn_gender, device, task=args.task)
        losses.append(loss)

        preds, labels = eval_loop(model, dev_loader, device)

        if args.task == 'age':
            cw = age_weights_metric
        elif args.task == 'gender':
            cw = gender_weights_metric
        else:
            cw = (age_weights_metric, gender_weights_metric)

        overall_acc, results = custom_metrics(preds, labels, args.task, cw)

        print(f"Epoch {epoch}: loss={loss:.4f} | weighted_acc={overall_acc:.4f}")

        if overall_acc > best_acc:
            best_acc = overall_acc
            patience_counter = 0
            best_results = results.copy()
            if args.save_model:
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.task}.pt"))
        elif args.early_stopping:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
        
    
    best_results["losses"] = losses.copy()
    with open(json_file, 'w') as jf:
        json.dump(best_results, jf, indent=2)

    return best_results

# ===== Main training with multiple runs =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',     type=str,   default='data_cslu_splits/gender/data/scripted')
    parser.add_argument('--output_dir',       type=str,   default='AgeGenderModels')
    parser.add_argument('--results_dir',      type=str,   default='results/classifier')
    parser.add_argument('--train_batch_size', type=int,   default=16)
    parser.add_argument('--eval_batch_size',  type=int,   default=32)
    parser.add_argument('--num_train_epochs', type=int,   default=50)
    parser.add_argument('--learning_rate',    type=float, default=1e-4)
    parser.add_argument('--task',             type=str,   choices=['age','gender','both'], default='both')
    parser.add_argument('--patience',         type=int,   default=10, help='Early stopping patience')
    parser.add_argument('--save_model',       action='store_true')
    parser.add_argument('--seed',             type=int,   default=0)
    parser.add_argument('--early_stopping',   action='store_true')
    parser.add_argument('--num_runs',         type=int,   default=2, help='Number of runs to average over')
    args = parser.parse_args()

    all_run_results = []
    single_run_losses = None

    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        set_seed(run_seed)
        print(f"\n=== Run {run_idx + 1}/{args.num_runs}, Seed: {run_seed} ===")
        run_results = train_age_gender_classifier(args)

        # Save the first run’s per-epoch losses for plotting
        if run_idx == 0 and 'losses' in run_results:
            single_run_losses = run_results['losses']

        all_run_results.append(run_results)

    # Aggregate metrics across runs (excluding 'losses')
    metric_keys = set()
    for res in all_run_results:
        metric_keys.update(res.keys())
    metric_keys.discard('losses')

    aggregated = {}
    for key in metric_keys:
        vals = [res[key] for res in all_run_results if key in res]
        aggregated[key + '_mean'] = float(np.mean(vals))
        aggregated[key + '_std']  = float(np.std(vals))

    # Include the first run’s losses separately
    if single_run_losses is not None:
        aggregated['losses_first_run'] = single_run_losses

    # Save aggregated results
    agg_file = os.path.join(args.results_dir, f"{args.task}_res_{args.num_runs}runs.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(agg_file, 'w') as af:
        json.dump(aggregated, af, indent=2)

    print(f"\nAggregated results saved to {agg_file}")
    print(json.dumps(aggregated, indent=2))

if __name__ == '__main__':
    main()