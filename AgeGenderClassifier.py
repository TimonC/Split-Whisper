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
    # unchanged
    ...

class AgeGenderCNN(nn.Module):
    # unchanged
    ...

# ===== Data Loading & Collate =====
# unchanged: combine_datasets, hf_collate_fn, train_loop, eval_loop
...  

# ===== Main Trainer =====
def train_age_gender_classifier(args):
    true_class_names = ["younger_Boy", "younger_Girl", "older_Boy", "older_Girl"]
    train_ds, dev_ds = combine_datasets(true_class_names, args.dataset_path)
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

    os.makedirs(os.path.join(args.output_dir, 'results', 'classifier'), exist_ok=True)
    json_file = os.path.join(args.output_dir, 'results', 'classifier', f"{args.model_name}.json")

    best_metric = 0.0
    patience_counter = 0

    for epoch in trange(args.num_train_epochs, desc='Epochs'):
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
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model_name}.pt"))

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
                break

# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data_cslu_splits/gender/data-timeframe/scripted')
    parser.add_argument('--output_dir', type=str, default='AgeGenderModels')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--task', type=str, choices=['age','gender','both'], default='both')
    parser.add_argument('--model_name', type=str, default='classifier')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='Whether to save the model checkpoints')
    args = parser.parse_args()
    train_age_gender_classifier(args)

if __name__ == '__main__':
    main()
