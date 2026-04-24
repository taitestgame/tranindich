"""
============================================================================
  BUOC 4: SU DUNG MODEL DICH  (translate.py)
============================================================================
  Load model fine-tuned (hoac goc) + cache thong minh.
  Sau khi chay 1 thoi gian, gan nhu KHONG can goi model nua.

  Su dung:
    python translate.py                    # Demo
    python translate.py -i                 # Tuong tac (go cau dich)
    python translate.py -t "Hello world"   # Dich 1 cau

  Import:
    from translate import Translator
    t = Translator()
    t.translate("Hello world")
    t.translate_batch(["Hello", "World"])
============================================================================
"""

import os
import re
import sys
import json
import time
import atexit
import argparse
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNED_DIR = os.path.join(BASE_DIR, "finetuned_model")
MODEL_LOCAL = os.path.join(BASE_DIR, "model_local")
CACHE_FILE = os.path.join(BASE_DIR, "cache.json")
MODEL_NAME = "Helsinki-NLP/opus-mt-en-vi"

# Toi uu CPU
MAX_LENGTH = 64
NUM_BEAMS = 2

# Cache
SAVE_EVERY = 10       # Ghi file sau N ban dich moi
FUZZY_THRESHOLD = 0.85  # Nguong fuzzy matching


class Translator:
    """
    English -> Vietnamese Translator.

    Uu tien:
      1. Exact cache   (0ms)
      2. Fuzzy cache   (~1ms)
      3. Model local   (~500-1500ms)
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._new = 0
        self._model = None
        self._tokenizer = None

        self._load_cache()
        self._load_model()
        atexit.register(self._save_cache)

        print(f"[Translator] Ready | Cache: {len(self._cache)} entries")

    # --- Cache ---

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except:
                self._cache = {}

    def _save_cache(self):
        if self._new > 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            self._new = 0

    def _auto_save(self):
        if self._new >= SAVE_EVERY:
            self._save_cache()

    # --- Model ---

    def _load_model(self):
        from transformers import MarianMTModel, MarianTokenizer

        # Uu tien: finetuned > local > HuggingFace
        if os.path.exists(FINETUNED_DIR):
            path = FINETUNED_DIR
        elif os.path.exists(MODEL_LOCAL):
            path = MODEL_LOCAL
        else:
            path = MODEL_NAME
        print(f"[Translator] Loading: {path}")

        self._tokenizer = MarianTokenizer.from_pretrained(path)
        self._model = MarianMTModel.from_pretrained(path)
        self._model.to("cpu")
        self._model.eval()

    # --- Normalize ---

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip().lower())

    # --- Fuzzy ---

    def _fuzzy(self, text: str) -> Optional[Tuple[str, str, float]]:
        best = None
        best_r = 0.0
        for k, v in self._cache.items():
            r = SequenceMatcher(None, text, k).ratio()
            if r > best_r:
                best_r = r
                best = (k, v, r)
        return best if best and best[2] >= FUZZY_THRESHOLD else None

    # --- Dich 1 cau ---

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        key = self.normalize(text)

        # 1. Exact cache
        if key in self._cache:
            return self._cache[key]

        # 2. Fuzzy
        f = self._fuzzy(key)
        if f:
            self._cache[key] = f[1]
            self._new += 1
            self._auto_save()
            return f[1]

        # 3. Model
        import torch
        inp = self._tokenizer(key, return_tensors="pt", padding=True,
                              truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            out = self._model.generate(**inp, max_length=MAX_LENGTH, num_beams=NUM_BEAMS)
        result = self._tokenizer.decode(out[0], skip_special_tokens=True)

        self._cache[key] = result
        self._new += 1
        self._auto_save()
        return result

    # --- Dich batch ---

    def translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        results = [""] * len(texts)
        need = []  # (idx, key)

        for i, t in enumerate(texts):
            if not t or not t.strip():
                continue
            key = self.normalize(t)
            if key in self._cache:
                results[i] = self._cache[key]
                continue
            f = self._fuzzy(key)
            if f:
                results[i] = f[1]
                self._cache[key] = f[1]
                self._new += 1
                continue
            need.append((i, key))

        if need:
            import torch
            indices = [x[0] for x in need]
            keys = [x[1] for x in need]

            inp = self._tokenizer(keys, return_tensors="pt", padding=True,
                                  truncation=True, max_length=MAX_LENGTH)
            with torch.no_grad():
                out = self._model.generate(**inp, max_length=MAX_LENGTH, num_beams=NUM_BEAMS)

            for j, idx in enumerate(indices):
                r = self._tokenizer.decode(out[j], skip_special_tokens=True)
                results[idx] = r
                self._cache[keys[j]] = r
                self._new += 1
            self._auto_save()

        return results

    # --- Utils ---

    def stats(self) -> Dict:
        return {"cache": len(self._cache), "pending": self._new}

    def save(self):
        self._new = max(self._new, 1)
        self._save_cache()


# ============================================================================
# TUONG TAC
# ============================================================================

def interactive(t: Translator):
    print("\n" + "=" * 55)
    print("  GO CAU TIENG ANH DE DICH | 'quit' de thoat")
    print("=" * 55 + "\n")

    while True:
        try:
            text = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        if text.lower() == "stats":
            print(t.stats())
            continue

        s = time.time()
        result = t.translate(text)
        ms = (time.time() - s) * 1000
        src = "CACHE" if ms < 10 else "MODEL"
        print(f"VI> {result}")
        print(f"    [{src}] {ms:.0f}ms\n")

    t.save()


# ============================================================================
# DEMO
# ============================================================================

def demo(t: Translator):
    print("\n" + "=" * 55)
    print("  DEMO DICH ANH -> VIET")
    print("=" * 55)

    sentences = [
        "Jupiter has just under 70 documented moons.",
        "The company decided to expand its operations into the European market.",
        "A new drug could help reduce damage to the body after a heart attack.",
        "The marketing team is working hard to finish the project.",
        "Fast food restaurants in the area are very popular with young people.",
    ]

    print("\n--- Dich don le ---")
    for s in sentences:
        t0 = time.time()
        r = t.translate(s)
        ms = (time.time() - t0) * 1000
        print(f"  EN: {s}")
        print(f"  VI: {r}  ({ms:.0f}ms)\n")

    print("--- Cache hit (dich lai) ---")
    t0 = time.time()
    for s in sentences:
        t.translate(s)
    ms = (time.time() - t0) * 1000
    print(f"  {len(sentences)} cau: {ms:.1f}ms (tu cache)\n")

    print("--- Batch ---")
    batch = [
        "We are pleased to announce that Mr. Kim has been promoted.",
        "The CEO gave a motivational speech at the meeting.",
        "If the shipment doesn't arrive by tomorrow, contact the supplier.",
    ]
    t0 = time.time()
    rs = t.translate_batch(batch)
    ms = (time.time() - t0) * 1000
    for en, vi in zip(batch, rs):
        print(f"  EN: {en}")
        print(f"  VI: {vi}\n")
    print(f"  Batch {len(batch)} cau: {ms:.0f}ms\n")

    print(f"Stats: {t.stats()}")
    t.save()
    print("\nHOAN TAT!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dich EN->VI")
    p.add_argument("-i", "--interactive", action="store_true")
    p.add_argument("-t", "--text", type=str)
    a = p.parse_args()

    tr = Translator()

    if a.text:
        print(tr.translate(a.text))
        tr.save()
    elif a.interactive:
        interactive(tr)
    else:
        demo(tr)
