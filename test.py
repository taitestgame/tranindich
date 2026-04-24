"""
============================================================================
  BUOC 3: TEST / DANH GIA MODEL  (test.py)
============================================================================
  - So sanh model goc vs model fine-tuned
  - Tinh BLEU score tren test set
  - In bang ket qua + vi du chi tiet

  Chay: python test.py
============================================================================
"""

import os
import sys
import json
import time
import math
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(BASE_DIR, "data", "test.json")
FINETUNED_DIR = os.path.join(BASE_DIR, "finetuned_model")
RESULT_FILE = os.path.join(BASE_DIR, "test_results.json")
MODEL_NAME = "Helsinki-NLP/opus-mt-en-vi"
MODEL_LOCAL = os.path.join(BASE_DIR, "model_local")

MAX_LENGTH = 128
NUM_BEAMS = 4


# ============================================================================
# BLEU SCORE (tu viet, khong can them thu vien)
# ============================================================================

def bleu_score(ref: str, hyp: str, max_n: int = 4) -> float:
    """Tinh BLEU score giua reference va hypothesis."""
    ref_tok = ref.lower().split()
    hyp_tok = hyp.lower().split()

    if not hyp_tok:
        return 0.0

    # Brevity penalty
    bp = 1.0
    if len(hyp_tok) < len(ref_tok):
        bp = math.exp(1 - len(ref_tok) / max(len(hyp_tok), 1))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(tuple(ref_tok[i:i+n]) for i in range(len(ref_tok)-n+1))
        hyp_ng = Counter(tuple(hyp_tok[i:i+n]) for i in range(len(hyp_tok)-n+1))
        clipped = sum(min(c, ref_ng.get(ng, 0)) for ng, c in hyp_ng.items())
        total = sum(hyp_ng.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_avg)


# ============================================================================
# DICH BATCH
# ============================================================================

def translate_batch(model, tokenizer, sentences, batch_size=16):
    """Dich danh sach cau."""
    import torch
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LENGTH
        )
        with torch.no_grad():
            out = model.generate(**inputs, max_length=MAX_LENGTH, num_beams=NUM_BEAMS)
        for j in range(len(batch)):
            results.append(tokenizer.decode(out[j], skip_special_tokens=True))
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    from transformers import MarianMTModel, MarianTokenizer

    print("=" * 65)
    print("  DANH GIA MODEL: Goc vs Fine-tuned")
    print("=" * 65)

    # 1. Load test data
    if not os.path.exists(TEST_FILE):
        print(f"[LOI] Khong tim thay {TEST_FILE}")
        print("Chay truoc: python prepare_data.py")
        sys.exit(1)

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    en_sents = [d["en"] for d in test_data]
    vi_refs = [d["vi"] for d in test_data]
    print(f"Test samples: {len(test_data)}")

    results = {}

    # 2. Model goc (local)
    model_path = MODEL_LOCAL if os.path.exists(MODEL_LOCAL) else MODEL_NAME
    print(f"\n--- MODEL GOC: {model_path} ---")
    tok1 = MarianTokenizer.from_pretrained(model_path)
    m1 = MarianMTModel.from_pretrained(model_path)
    m1.to("cpu"); m1.eval()

    t = time.time()
    base_trans = translate_batch(m1, tok1, en_sents)
    base_time = time.time() - t
    del m1

    base_bleu = [bleu_score(r, h) for r, h in zip(vi_refs, base_trans)]
    avg_base = sum(base_bleu) / len(base_bleu)
    results["base"] = {"BLEU-4": round(avg_base, 4), "time_s": round(base_time, 1)}
    print(f"  BLEU-4: {avg_base:.4f} | Time: {base_time:.1f}s")

    # 3. Model fine-tuned
    ft_trans = None
    if os.path.exists(FINETUNED_DIR):
        print(f"\n--- MODEL FINE-TUNED: {FINETUNED_DIR} ---")
        tok2 = MarianTokenizer.from_pretrained(FINETUNED_DIR)
        m2 = MarianMTModel.from_pretrained(FINETUNED_DIR)
        m2.to("cpu"); m2.eval()

        t = time.time()
        ft_trans = translate_batch(m2, tok2, en_sents)
        ft_time = time.time() - t
        del m2

        ft_bleu = [bleu_score(r, h) for r, h in zip(vi_refs, ft_trans)]
        avg_ft = sum(ft_bleu) / len(ft_bleu)
        results["finetuned"] = {"BLEU-4": round(avg_ft, 4), "time_s": round(ft_time, 1)}
        print(f"  BLEU-4: {avg_ft:.4f} | Time: {ft_time:.1f}s")
    else:
        print(f"\n[WARN] Chua co model fine-tuned. Chay: python train.py")

    # 4. Bang so sanh
    print(f"\n{'='*65}")
    print("  BANG SO SANH")
    print(f"{'='*65}")

    if ft_trans:
        diff = avg_ft - avg_base
        sign = "+" if diff >= 0 else ""
        print(f"  {'':12} {'Goc':>12} {'Fine-tuned':>12} {'Chenh lech':>12}")
        print(f"  {'-'*48}")
        print(f"  {'BLEU-4':12} {avg_base:>12.4f} {avg_ft:>12.4f} {sign}{diff:>11.4f}")
        print(f"  {'Time(s)':12} {base_time:>12.1f} {ft_time:>12.1f}")
    else:
        print(f"  BLEU-4 (goc): {avg_base:.4f}")

    # 5. Vi du
    print(f"\n{'='*65}")
    print("  VI DU DICH (10 cau)")
    print(f"{'='*65}")

    for i in range(min(10, len(test_data))):
        print(f"\n  --- Cau {i+1} ---")
        print(f"  EN:   {en_sents[i][:75]}")
        print(f"  Ref:  {vi_refs[i][:75]}")
        print(f"  Base: {base_trans[i][:75]}")
        if ft_trans:
            print(f"  FT:   {ft_trans[i][:75]}")
        b = bleu_score(vi_refs[i], base_trans[i])
        print(f"  BLEU: {b:.4f}", end="")
        if ft_trans:
            fb = bleu_score(vi_refs[i], ft_trans[i])
            print(f" -> {fb:.4f}", end="")
        print()

    # 6. Luu
    detailed = []
    for i in range(len(test_data)):
        d = {
            "en": en_sents[i], "vi_ref": vi_refs[i],
            "base": base_trans[i],
            "base_bleu": round(bleu_score(vi_refs[i], base_trans[i]), 4),
        }
        if ft_trans:
            d["ft"] = ft_trans[i]
            d["ft_bleu"] = round(bleu_score(vi_refs[i], ft_trans[i]), 4)
        detailed.append(d)

    results["detailed"] = detailed
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  Ket qua luu: {RESULT_FILE}")
    print(f"  Dung model: python translate.py")


if __name__ == "__main__":
    main()
