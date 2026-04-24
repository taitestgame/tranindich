"""
Tai model Helsinki-NLP/opus-mt-en-vi ve luu local.
Chay 1 lan duy nhat.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "model_local")
MODEL_NAME = "Helsinki-NLP/opus-mt-en-vi"

def main():
    from transformers import MarianMTModel, MarianTokenizer

    print(f"Dang tai model: {MODEL_NAME}")
    print(f"Luu vao: {SAVE_DIR}")
    print("-" * 50)

    print("[1/2] Tai tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

    print("[2/2] Tai model...")
    model = MarianMTModel.from_pretrained(MODEL_NAME)

    # Luu
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    print(f"\nHOAN TAT! Model da luu tai: {SAVE_DIR}")

    # Thu dich 1 cau de test
    import torch
    model.to("cpu")
    model.eval()
    test = "The company decided to expand its operations."
    inp = tokenizer(test, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model.generate(**inp, max_length=64, num_beams=2)
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\nTest nhanh:")
    print(f"  EN: {test}")
    print(f"  VI: {result}")

if __name__ == "__main__":
    main()
