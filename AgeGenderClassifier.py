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

class AgeGenderCNN(nn.Module):
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
        self.shared_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        # Define heads
        if self.task in ('age', 'both'):
            self.age_head = nn.Linear(64, 1)
        if self.task in ('gender', 'both'):
            self.gender_head = nn.Linear(64, 1)

    def _make_layer(self, in_c, out_c, stride=1):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        return nn.Sequential(
            ResidualBlock(in_c, out_c, stride, downsample),
            ResidualBlock(out_c, out_c)
        )

    def forward(self, x, mask=None):
        # x: (N,1,H,W), mask unused here
        x = self.input_norm(x)
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.shared_fc(x)

        outputs = {}
        if self.task in ('age', 'both'):
            outputs['age'] = self.age_head(x).squeeze(1)
        if self.task in ('gender', 'both'):
            outputs['gender'] = self.gender_head(x).squeeze(1)
        return outputs

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

# ===== Main Trainer =====
def train_age_gender_classifier(args):
    # Determine class names based on task
    train_ds, dev_ds = combine_datasets(args.dataset_path)
    for d in [train_ds, dev_ds]:
        d.set_format(type='torch', columns=['input_features', 'y_age', 'y_gender'])
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=hf_collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=hf_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AgeGenderCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    # Prepare results structure
    max_epochs = args.num_train_epochs
    results = {
        'epoch': list(range(1, max_epochs+1)),
        'loss': [None] * max_epochs,
        'accuracy_age': [None] * max_epochs,
        'accuracy_gender': [None] * max_epochs,
        'accuracy_joint': [None] * max_epochs,
        'f1_age': [None] * max_epochs,
        'f1_gender': [None] * max_epochs,
        'f1_joint': [None] * max_epochs
    }

    results_dir = os.makedirs(os.path.join(args.output_dir, 'results', 'classifier'), exist_ok=True)
    json_file = os.path.join(results_dir, f"{args.task}.json")

    best_metric = 0.0
    patience_counter = 0

    for epoch in trange(max_epochs, desc='Epochs'):
        idx = epoch
        train_loss = train_loop(model, train_loader, optimizer, loss_fn, device, args.task)
        preds, labels = eval_loop(model, dev_loader, device, args.task)

        # compute metrics
        acc_age = f1_age = acc_gen = f1_gen = acc_joint = f1_joint = None
        if 'age' in preds:
            acc_age = accuracy_score(labels['age'], preds['age'])
            f1_age = f1_score(labels['age'], preds['age'])
        if 'gender' in preds:
            acc_gen = accuracy_score(labels['gender'], preds['gender'])
            f1_gen = f1_score(labels['gender'], preds['gender'])
        if 'joint' in preds:
            acc_joint = accuracy_score(labels['joint'], preds['joint'])
            f1_joint = f1_score(labels['joint'], preds['joint'], average='weighted')

        # store
        results['loss'][idx] = train_loss
        results['accuracy_age'][idx] = acc_age
        results['accuracy_gender'][idx] = acc_gen
        results['accuracy_joint'][idx] = acc_joint
        results['f1_age'][idx] = f1_age
        results['f1_gender'][idx] = f1_gen
        results['f1_joint'][idx] = f1_joint

        # conditional model save
        if args.save_model:
            metric = acc_joint if args.task=='both' else (acc_age if args.task=='age' else acc_gen)
            if metric and metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.task}.pt"))

        # nicely formatted print
        print(
            f"Epoch {epoch+1}/{max_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Age: Acc={acc_age:.3f if acc_age is not None else 'N/A'} F1={f1_age:.3f if f1_age is not None else 'N/A'} | "
            f"Gender: Acc={acc_gen:.3f if acc_gen is not None else 'N/A'} F1={f1_gen:.3f if f1_gen is not None else 'N/A'} | "
            f"Joint: Acc={acc_joint:.3f if acc_joint is not None else 'N/A'} F1={f1_joint:.3f if f1_joint is not None else 'N/A'}"
        )

        # save results JSON each epoch
        with open(json_file, 'w') as jf:
            json.dump(results, jf, indent=2)

        # scheduler and early stopping
        metric = acc_joint if args.task=='both' else (acc_age if args.task=='age' else acc_gen)
        scheduler.step(metric)
        if metric and metric > best_metric:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data_cslu_splits/gender/data/scripted')
    parser.add_argument('--output_dir', type=str, default='AgeGenderModels')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--task', type=str, choices=['age','gender','both'], default='both')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_model', action='store_true', default=False)
    args = parser.parse_args()
    train_age_gender_classifier(args)

if __name__ == '__main__':
    main()
