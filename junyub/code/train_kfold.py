import os
import gc
import json

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from dataset import DocDataset, MultiRecipeDataset, get_train_recipes

SEED = 42
IMG_SIZE = 384
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FULL = 15
FOLDS = 5
MODEL_NAME = 'convnext_tiny'
NUM_WORKERS = 4

DATA_DIR = '../data'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
META_CSV = os.path.join(DATA_DIR, 'meta.csv')
OUT_DIR = '../results'
os.makedirs(OUT_DIR, exist_ok=True)

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, targets):
        logp = nn.functional.log_softmax(logits, dim=1)
        ce = nn.functional.nll_loss(logp, targets, weight=self.ce.weight, reduction='none')
        p = torch.exp(-ce)
        loss = ((1-p) ** self.gamma) * ce
        return loss.mean()

def build_model(num_classes):
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    return model

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    all_preds, all_targets = [], []

    for imgs, targets in tqdm(loader, leave=False):
        if isinstance(imgs, torch.Tensor):
            imgs_dev = imgs.to(device, non_blocking=True)
            logits = model(imgs_dev)
        else:
            logits_list = []
            for x in imgs:
                x = x.to(device, non_blocking=True)
                logits_list.append(model(x))
            logits = torch.stack(logits_list, dim=0).mean(0)

        targets = targets.to(device, non_blocking=True)

        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_preds.extend(logits.detach().softmax(1).cpu().numpy().argmax(1))
        all_targets.extend(targets.detach().cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average='macro')
    return np.mean(losses), f1


@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_targets = [], []

    for imgs, targets in loader:
        if isinstance(imgs, torch.Tensor):
            imgs_dev = imgs.to(device, non_blocking=True)
            logits = model(imgs_dev)
        else:
            logits_list = []
            for x in imgs:
                x = x.to(device, non_blocking=True)
                logits_list.append(model(x))
            logits = torch.stack(logits_list, dim=0).mean(0)

        targets = targets.to(device, non_blocking=True)
        loss = criterion(logits, targets)
        losses.append(loss.item())

        all_preds.extend(logits.softmax(1).cpu().numpy().argmax(1))
        all_targets.extend(targets.detach().cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average='macro')
    return np.mean(losses), f1


def get_class_weights(train_df, num_classes):
    counts = train_df['target'].value_counts().reindex(range(num_classes), fill_value=0).values.astype(np.float32)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float)

def to_device_views(imgs, device):
    """
    imgs가 Tensor면 그대로, list[Tensor]면 각 텐서를 device로 옮겨 반환.
    반환 형태는 입력 형태를 그대로 유지 (Tensor 또는 list[Tensor]).
    """
    if isinstance(imgs, torch.Tensor):
        return imgs.to(device, non_blocking=True)
    elif isinstance(imgs, (list, tuple)):
        return [x.to(device, non_blocking=True) for x in imgs]
    else:
        raise TypeError(f'Unexpected imgs type: {type(imgs)}')

@torch.no_grad()
def forward_views(model, imgs, device):
    """
    imgs가 Tensor: model(imgs) 반환
    imgs가 list[Tensor]: 각 뷰별 forward 후 평균(logits 평균) 반환
    """
    if isinstance(imgs, torch.Tensor):
        return model(imgs.to(device, non_blocking=True)).softmax(1).cpu()
    elif isinstance(imgs, (list, tuple)):
        logits_list = []
        for x in imgs:
            logit = model(x.to(device, non_blocking=True))
            logits_list.append(logit)
        logits = torch.stack(logits_list, dim=0).mean(0)
        return logits.softmax(1).cpu()
    else:
        raise TypeError(f'Unexpected imgs type: {type(imgs)}')


def main():
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_df = pd.read_csv(TRAIN_CSV)
    meta_df = pd.read_csv(META_CSV)
    num_classes = train_df['target'].nunique()


    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(train_df), num_classes), dtype=np.float32)
    oof_targets = train_df['target'].values

    class_weight = get_class_weights(train_df, num_classes).to(device)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, train_df['target'])):
        print(f'\n=============== Fold {fold} ===============')
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        sample_weights = class_weight.detach().cpu().numpy()[tr_df['target'].values]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_recipes = get_train_recipes(IMG_SIZE)
        train_ds = MultiRecipeDataset(tr_df, TRAIN_IMG_DIR, label_col='target', img_size=IMG_SIZE, recipes=train_recipes, is_test=False)

        val_ds = MultiRecipeDataset(va_df, TRAIN_IMG_DIR, label_col='target', img_size=IMG_SIZE, recipes=None, is_test=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model = build_model(num_classes).to(device)

        for n, p in model.named_parameters():
            p.requires_grad = ('head' in n) or ('fc' in n) or ('classifier' in n)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)
        criterion = FocalLoss(gamma=1.5, weight=class_weight)

        best_f1, best_path = -1, os.path.join(OUT_DIR, f'{MODEL_NAME}_fold{fold}.pt')

        for epoch in range(1, EPOCHS_HEAD+1):
            tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_f1 = valid_one_epoch(model, val_loader, criterion, device)
            print(f'[Head][{epoch}/{EPOCHS_HEAD}] tr_loss:{tr_loss:.4f} tr_f1:{tr_f1:.4f} | va_loss:{va_loss:.4f} va_f1:{va_f1:.4f}')
            if va_f1 > best_f1:
                best_f1 = va_f1
                torch.save(model.state_dict(), best_path)

        for p in model.parameters(): p.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        patience, patience_cnt = 4, 0
        for epoch in range(1, EPOCHS_FULL+1):
            tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_f1 = valid_one_epoch(model, val_loader, criterion, device)
            print(f'[Full][{epoch}/{EPOCHS_FULL}] tr_loss:{tr_loss:.4f} tr_f1:{tr_f1:.4f} | va_loss:{va_loss:.4f} va_f1:{va_f1:.4f}')
            if va_f1 > best_f1:
                best_f1 = va_f1
                torch.save(model.state_dict(), best_path)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print('Early stopping!')
                    break

        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        preds = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                logits = forward_views(model, imgs, device)
                preds.append(logits.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        oof_preds[va_idx] = preds

        print(f'Fold {fold} best Macro-F1: {best_f1:.4f}')

        del model, train_ds, val_ds, train_loader, val_loader, preds
        gc.collect(); torch.cuda.empty_cache()

    oof_labels = oof_preds.argmax(1)
    oof_f1 = f1_score(oof_targets, oof_labels, average='macro')
    print(f'\nOOF Macro-F1: {oof_f1:.4f}')
    np.save(os.path.join(OUT_DIR, f'{MODEL_NAME}_oof_probs.npy'), oof_preds)

if __name__ == '__main__':
    main()