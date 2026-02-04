import os
import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaForSequenceClassification, PreTrainedTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset

# --- CONFIGURATION ---
# We define where we expect Member C to put the data.
# Even if it's not there yet, we hardcode the path now.
DATA_PATH = "../data/mixed_security_dataset.csv"
TOKENIZER_DIR = "./tokenizer"
MODEL_DIR = "./models"
ONNX_PATH = "./models/neurowall.onnx"

def train_pipeline():
    # --- STEP 1: SAFETY CHECK ---
    # You are checking if the fuel (data) exists before starting the engine.
    if not os.path.exists(DATA_PATH):
        print(f"❌ Waiting for data... File not found at {DATA_PATH}")
        print("💡 TIP: Creates a dummy file to test your code? (See Phase 3 below)")
        return

    print("🚀 Loading Data...")
    # usage of astype(str) is crucial. Hackers send "NaN" or numbers to crash parsers.
    # We force everything to be a string to prevent crashes.
    df = pd.read_csv(DATA_PATH).astype(str)
    
    # --- STEP 2: TEACHING THE VOCABULARY (Tokenizer) ---
    print("🧠 Training Custom Tokenizer...")
    # Standard BERT knows English ("Apple", "Run"). 
    # It does NOT know SQL ("SELECT", "UNION", "1=1").
    # We dump all our data to a text file so the Tokenizer can learn these specific jargon words.
    with open("temp_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(df['text'].tolist()))

    # We use Byte-Level BPE. This creates tokens from bytes, not just words.
    # This allows the model to read "Hex Code" attacks that look like garbage text.
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=["temp_corpus.txt"],
        vocab_size=30_000, # 30k words is the standard vocabulary size
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask,"]
    )
    
    # Save this "dictionary" so Member A can use it later to translate user input.
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_model(TOKENIZER_DIR)
    
    # Wrap it in a wrapper that PyTorch understands
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        vocab_file=f"{TOKENIZER_DIR}/vocab.json",
        merges_file=f"{TOKENIZER_DIR}/merges.txt",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<s>",
        sep_token="</s>",
        mask_token="<mask,>"
    )
    fast_tokenizer.save_pretrained(TOKENIZER_DIR)

    # --- STEP 3: PREPARING THE BRAIN (Model Config) ---
    print("🏋️ Initializing DistilRoBERTa Configuration...")
    
    # 1. Tokenize the data (Convert text "SELECT" -> Number [492, 102])
    dataset = Dataset.from_pandas(df)
    def tokenize_function(examples):
        return fast_tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128 # We cap input at 128 tokens to keep it fast.
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 2. Define the Architecture
    # CRITICAL: We are using only 2 layers (num_hidden_layers=2).
    # Standard BERT has 12 layers. 
    # By using 2, we sacrifice ~2% accuracy for 500% speed increase.
    config = RoBERTaConfig(
        vocab_size=30_000,
        max_position_embeddings=130,
        num_attention_heads=4,
        num_hidden_layers=2, 
        type_vocab_size=1,
    )
    
    model = RoBERTaForSequenceClassification(config=config, num_labels=2) # 0=Safe, 1=Bad

    # --- STEP 4: THE GYM (Training Loop) ---
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1, # One pass is usually enough for distinct patterns like SQLi
        per_device_train_batch_size=16,
        logging_steps=10,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets, # We skip validation split for the 'Dry Run'
        tokenizer=fast_tokenizer,
    )

    trainer.train()

    # --- STEP 5: THE EXPORT (ONNX) ---
    print("📦 Exporting to ONNX...")
    model.eval() # Switch from "Learning Mode" to "Testing Mode"
    
    # We create a fake input ("dummy") to show the ONNX exporter 
    # what the data shape looks like. It traces the math operations.
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
    
    # Clean up temp file
    if os.path.exists("temp_corpus.txt"):
        os.remove("temp_corpus.txt")

if __name__ == "__main__":
    train_pipeline()