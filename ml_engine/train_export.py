import os
import shutil
import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset

# --- CONFIGURATION ---
DATA_PATH = "../data/mixed_security_dataset.csv"
TOKENIZER_DIR = "./tokenizer"
MODEL_DIR = "./models"
BACKUP_DIR = "./saved_model_backup" # <--- NEW SAFETY BACKUP
ONNX_PATH = "./models/neurowall.onnx"

def train_pipeline():
    # --- STEP 1: LOAD DATA ---
    if not os.path.exists(DATA_PATH):
        print(f"❌ Waiting for data... File not found at {DATA_PATH}")
        return

    print("🚀 Loading Data...")
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].astype(str).fillna("")
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    # --- STEP 2: TRAIN TOKENIZER ---
    print("🧠 Training Custom Tokenizer...")
    with open("temp_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(df['text'].tolist()))

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=["temp_corpus.txt"],
        vocab_size=30_000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask,>"]
    )
    
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_model(TOKENIZER_DIR)
    
    print("🔄 Reloading Tokenizer via RobertaTokenizerFast...")
    fast_tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_DIR, max_len=128)

    # --- STEP 3: PREPARE DATASET ---
    print("🏋️ Tokenizing Data...")
    dataset = Dataset.from_pandas(df)
    
    def tokenize_function(examples):
        return fast_tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # --- STEP 4: CONFIGURE & TRAIN MODEL ---
    print("🔧 Initializing Model...")
    
    config = RobertaConfig(
        vocab_size=30_000,
        max_position_embeddings=130,
        num_attention_heads=4,
        num_hidden_layers=2, 
        type_vocab_size=1,
        num_labels=2 
    )
    
    model = RobertaForSequenceClassification(config)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        logging_steps=50,
        learning_rate=5e-5,
        save_strategy="no",
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    print("🏃 Starting Training...")
    trainer.train()

    # --- NEW STEP: SAFETY SAVE ---
    print("💾 Saving Backup Model to Disk (Safety First!)...")
    trainer.save_model(BACKUP_DIR)
    print("✅ Backup Saved.")

    # --- STEP 5: EXPORT TO ONNX ---
    print("📦 Exporting to ONNX...")
    model.eval()
    
    dummy_input = fast_tokenizer("SELECT * FROM users", return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.onnx.export(
        model, 
        (dummy_input['input_ids'], dummy_input['attention_mask']), 
        ONNX_PATH,
        input_names=['input_ids', 'attention_mask'], 
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'}},
        opset_version=14
    )
    
    print(f"🎉 SUCCESS! Brain saved to {ONNX_PATH}")
    
    if os.path.exists("temp_corpus.txt"):
        os.remove("temp_corpus.txt")

if __name__ == "__main__":
    train_pipeline()