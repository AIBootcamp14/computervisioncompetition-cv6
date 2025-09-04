import os
import glob

import timm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MultiRecipeDataset

IMG_SIZE = 384
BATCH_SIZE = 32
MODEL_NAME = 'convnext_tiny'
NUM_WORKERS = 4

DATA_DIR = '../data'
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')
SUBMIT_PATH = f'../results/{MODEL_NAME}.csv'

WEIGHTS_GLOB = f'../results/{MODEL_NAME}_fold*.pt'

def build_model(num_classes):
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    return model

@torch.no_grad()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    meta_df = pd.read_csv(os.path.join(DATA_DIR, 'meta.csv'))
    num_classes = meta_df['target'].nunique()

    test_files = sorted([f for f in os.listdir(TEST_IMG_DIR)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    test_ids = [f for f in test_files]
    test_df = pd.DataFrame({'ID': test_ids})

    test_ds = MultiRecipeDataset(
        test_df, TEST_IMG_DIR, label_col=None, img_size=IMG_SIZE, recipes=None, is_test=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    weight_paths = sorted(glob.glob(WEIGHTS_GLOB))
    assert len(weight_paths) > 0, '가중치(.pt) 파일을 찾을 수 없습니다.'

    logits_sum = None
    all_ids = None

    for w in weight_paths:
        model = build_model(num_classes).to(device)
        state = torch.load(w, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()

        fold_logits = []
        fold_ids = []

        for imgs, ids in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs).detach().cpu().numpy()
            fold_logits.append(logits)
            fold_ids.extend(ids)

        fold_logits = np.concatenate(fold_logits, axis=0)

        if logits_sum is None:
            logits_sum = fold_logits
            all_ids = np.array(fold_ids)
        else:
            logits_sum += fold_logits

        del model

    probs = torch.tensor(logits_sum / len(weight_paths)).softmax(1).numpy()
    preds = probs.argmax(1)

    sub = pd.DataFrame({'ID': all_ids, 'target': preds})
    sub.to_csv(SUBMIT_PATH, index=False)
    print(f'Saved: {SUBMIT_PATH}')

if __name__ == '__main__':
    main()