"""
============================================================================
  BUOC 1: CHUAN BI DU LIEU  (prepare_data.py)
============================================================================
  - Doc tat ca file .txt TOEIC
  - Trich xuat cau tieng Anh
  - Dung Google Translate API (mien phi) de dich -> tao cap (en, vi)
  - Chia train/test (80/20)
  - Luu ra data/train.json va data/test.json

  Chay: python prepare_data.py
============================================================================
"""

import os
import re
import json
import glob
import time
import random
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.json")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.json")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ============================================================================
# TRICH XUAT CAU TIENG ANH TU FILE TXT
# ============================================================================

def extract_sentences(filepath: str) -> List[str]:
    """Trich xuat cau tieng Anh tu 1 file TOEIC .txt"""
    sentences = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"  [LOI] {filepath}: {e}")
        return []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Pattern: "Cau 101: ..." hoac "Cau 101 (Test 2): ..."
        m = re.match(r'C\u00e2u\s+\d+(?:\s*\(Test\s*\d+\))?:\s*(.+)', line)
        if m:
            eng = m.group(1).strip()
            # Cat phan dap an neu cung dong
            eng = re.split(r'\s+[A-D]\.\s', eng, maxsplit=1)[0].strip()
            eng = re.sub(r'\.{2,}', '___', eng)
            eng = re.sub(r'\(D\u1ef1a tr\u00ean.*?\)', '', eng)
            eng = re.sub(r'\(N\u1ed9i dung.*?\)', '', eng)
            if len(re.findall(r'[a-zA-Z]+', eng)) >= 4:
                sentences.append(eng.strip())
            continue

        # Dap an dang cau dai
        m2 = re.match(r'^[A-D]\.\s+(.{25,})', line)
        if m2:
            opt = m2.group(1).strip()
            opt = re.sub(r'\s*\u0110\u00e1p \u00e1n:\s*[A-D]\s*$', '', opt)
            if len(re.findall(r'[a-zA-Z]+', opt)) >= 5:
                sentences.append(opt)

    return sentences


def collect_all(txt_dir: str) -> List[str]:
    """Thu thap tat ca cau tieng Anh, loai trung."""
    all_sents = []
    files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    print(f"\nTim thay {len(files)} file .txt:")
    print("-" * 50)
    for f in files:
        sents = extract_sentences(f)
        all_sents.extend(sents)
        print(f"  {os.path.basename(f):20s} -> {len(sents):4d} cau")
    
    before = len(all_sents)
    all_sents = list(set(all_sents))
    print(f"-" * 50)
    print(f"  Tong: {before} | Unique: {len(all_sents)} | Trung: {before - len(all_sents)}")
    return sorted(all_sents)


# ============================================================================
# DICH BANG GOOGLE TRANSLATE API (MIEN PHI)
# ============================================================================

def translate_with_api(sentences: List[str], batch_size: int = 20) -> List[dict]:
    """
    Dung Google Translate (mien phi qua deep-translator)
    de dich tat ca cau tieng Anh sang tieng Viet.
    
    Day la buoc TAO DU LIEU CHAT LUONG CAO de training.
    Sau khi train xong, se KHONG can API nua.
    """
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source='en', target='vi')
    pairs = []
    total = len(sentences)
    errors = 0

    print(f"\nDang dich {total} cau bang Google Translate API...")
    print(f"(De tao du lieu training chat luong cao)")
    print("-" * 50)

    for i in range(0, total, batch_size):
        batch = sentences[i:i + batch_size]
        
        for src in batch:
            try:
                tgt = translator.translate(src)
                if tgt:
                    pairs.append({"en": src, "vi": tgt})
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [WARN] Loi dich: {str(e)[:60]}")
                # Doi 1 chut de tranh rate limit
                time.sleep(1)
        
        done = min(i + batch_size, total)
        pct = done / total * 100
        print(f"  [{done:4d}/{total:4d}] {pct:5.1f}% | OK: {len(pairs)} | Errors: {errors}")
        
        # Delay nhe de tranh rate-limit
        time.sleep(0.5)

    print(f"\nHoan tat! {len(pairs)}/{total} cau dich thanh cong")
    if errors > 0:
        print(f"  {errors} cau bi loi (da bo qua)")
    
    return pairs


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  CHUAN BI DU LIEU TRAINING")
    print("  Google Translate API -> Train data")
    print("=" * 60)

    # 1. Trich xuat cau
    sentences = collect_all(BASE_DIR)
    if not sentences:
        print("[LOI] Khong tim thay cau tieng Anh nao!")
        return

    # 2. Dich bang API
    pairs = translate_with_api(sentences)
    if not pairs:
        print("[LOI] Khong dich duoc cau nao!")
        return

    # 3. Chia train/test
    random.seed(RANDOM_SEED)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * TRAIN_RATIO)
    train_data = shuffled[:split]
    test_data = shuffled[split:]

    # 4. Luu file
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"\nDa luu:")
    print(f"  Train: {len(train_data)} cap -> {TRAIN_FILE}")
    print(f"  Test:  {len(test_data)} cap -> {TEST_FILE}")

    # Hien vi du
    print(f"\nVi du du lieu:")
    print("-" * 50)
    for p in pairs[:5]:
        print(f"  EN: {p['en'][:70]}")
        print(f"  VI: {p['vi'][:70]}")
        print()

    print("HOAN TAT! Chay tiep: python train.py")


if __name__ == "__main__":
    main()
