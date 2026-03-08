import torch
import os
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

# --- CONFIGURATION ---
BACKUP_DIR = "./saved_model_backup"
TOKENIZER_DIR = "./tokenizer"
MODEL_DIR = "./models"
ONNX_PATH = "./models/neurowall_v1_tiny.onnx"

def export_backup():
    print(f"🚀 Loading Brain from Backup: {BACKUP_DIR}...")
    
    # 1. Load the PyTorch Model from the backup
    try:
        model = RobertaForSequenceClassification.from_pretrained(BACKUP_DIR)
        tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_DIR, max_len=128)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading backup: {e}")
        return

    # 2. Prepare Dummy Input for Export
    print("📦 converting to ONNX...")
    model.eval()
    dummy_input = tokenizer("SELECT * FROM users", return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    
    # 3. Export
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        torch.onnx.export(
            model, 
            (dummy_input['input_ids'], dummy_input['attention_mask']), 
            ONNX_PATH,
            input_names=['input_ids', 'attention_mask'], 
            output_names=['logits'],
            dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'}},
            opset_version=18 # We use 18 because it's stable for new PyTorch
        )
        print(f"🎉 SUCCESS! Brain saved to {ONNX_PATH}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    export_backup()