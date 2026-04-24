"""
============================================================================
  BUOC 2: TRAINING MODEL  (train.py)
============================================================================
  - Load du lieu tu data/train.json (tao boi prepare_data.py)
  - Download model goc Helsinki-NLP/opus-mt-en-vi
  - Fine-tune tren du lieu TOEIC (API-generated)
  - Luu model fine-tuned ra finetuned_model/

  Chay: python train.py
  Tuy chinh: python train.py --epochs 5 --batch_size 8
============================================================================
"""

import os
import sys
import json
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
FINETUNED_DIR = os.path.join(BASE_DIR, "finetuned_model")
LOG_FILE = os.path.join(BASE_DIR, "training_log.json")

MODEL_NAME = "Helsinki-NLP/opus-mt-en-vi"
MODEL_LOCAL = os.path.join(BASE_DIR, "model_local")

# Mac dinh (toi uu cho CPU i5)
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 5e-5
DEFAULT_MAX_LEN = 128


# ============================================================================
# DATASET
# ============================================================================

class PairDataset:
    """Dataset cac cap (en, vi) da chuan bi."""

    def __init__(self, pairs, tokenizer, max_len=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        import torch
        p = self.pairs[idx]

        src = self.tokenizer(
            p["en"], max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        tgt = self.tokenizer(
            text_target=p["vi"], max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        input_ids = src["input_ids"].squeeze(0)
        attention_mask = src["attention_mask"].squeeze(0)
        labels = tgt["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================================
# TRAINING
# ============================================================================

def train(epochs, batch_size, lr, max_len):
    import torch
    from torch.utils.data import DataLoader
    from transformers import MarianMTModel, MarianTokenizer

    # 1. Load data
    if not os.path.exists(TRAIN_FILE):
        print(f"[LOI] Khong tim thay {TRAIN_FILE}")
        print("Chay truoc: python prepare_data.py")
        sys.exit(1)

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} cap training")

    # 2. Load model (uu tien local)
    model_path = MODEL_LOCAL if os.path.exists(MODEL_LOCAL) else MODEL_NAME
    print(f"\nLoad model: {model_path}")
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    model.to("cpu")
    model.train()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # 3. Dataset + DataLoader
    dataset = PairDataset(train_data, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_batches = len(loader)

    print(f"\nTraining config:")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  Max length: {max_len}")
    print(f"  Batches:    {total_batches}/epoch")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 5. Training loop
    history = []
    best_loss = float("inf")
    t0 = time.time()

    print(f"\n{'='*50}")
    print(f"  BAT DAU TRAINING")
    print(f"{'='*50}\n")

    from tqdm import tqdm

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        et = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", file=sys.stdout)
        for step, batch in enumerate(pbar, 1):
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            pbar.set_postfix({"Loss": f"{epoch_loss / step:.4f}"})

        avg_loss = epoch_loss / total_batches
        etime = time.time() - et
        history.append({
            "epoch": epoch, "loss": round(avg_loss, 4),
            "time_s": round(etime, 1)
        })

        print(f"  >>> Epoch {epoch}: Loss={avg_loss:.4f} | Time={etime:.1f}s\n")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(FINETUNED_DIR, exist_ok=True)
            model.save_pretrained(FINETUNED_DIR)
            tokenizer.save_pretrained(FINETUNED_DIR)
            print(f"  [SAVED] Best model -> {FINETUNED_DIR}\n")

    total_time = time.time() - t0

    # 6. Log
    log = {
        "model": MODEL_NAME,
        "train_samples": len(train_data),
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "best_loss": round(best_loss, 4),
        "total_time_s": round(total_time, 1),
        "history": history,
    }
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"{'='*50}")
    print(f"  TRAINING HOAN TAT!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Time:      {total_time:.1f}s")
    print(f"  Model:     {FINETUNED_DIR}")
    print(f"  Log:       {LOG_FILE}")
    print(f"{'='*50}")
    print(f"\nChay tiep: python test.py")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune MarianMT en->vi")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    a = p.parse_args()
    train(a.epochs, a.batch_size, a.lr, a.max_len)
