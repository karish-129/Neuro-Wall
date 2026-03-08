import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

# Paths
BACKUP_DIR = "./saved_model_backup"
TOKENIZER_DIR = "./tokenizer"

# Decision threshold
THRESHOLD = 0.4

def test_backup():
    print(f"🔎 Loading PyTorch Backup from {BACKUP_DIR}...")

    # 1. Load model & tokenizer
    try:
        model = RobertaForSequenceClassification.from_pretrained(BACKUP_DIR)
        tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_DIR)
        print("✅ Backup Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load backup: {e}")
        print("💡 Check if model + tokenizer folders exist.")
        return

    # 2. Device handling (CPU / GPU safe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Test HTTP-style input
    payload = "UNION SELECT * FROM users"
    full_request = (
        f"GET http://localhost:8080/search?q={payload} HTTP/1.1\n"
        f"User-Agent: Mozilla/5.0\n"
        f"Cookie: SESSION=123\n"
    )

    # 4. Tokenize
    inputs = tokenizer(
        full_request,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 5. Inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    safe_prob = probs[0][0].item()
    attack_prob = probs[0][1].item()

    # 6. Output
    print(f"\n📝 Payload: {payload}")
    print(f"🟢 Safe Confidence:   {safe_prob*100:.1f}%")
    print(f"🔴 Attack Confidence: {attack_prob*100:.1f}%")

    if attack_prob >= THRESHOLD:
        print("🛡️  RESULT: BLOCKED")
    else:
        print("✅ RESULT: ALLOWED")

if __name__ == "__main__":
    test_backup()
