import onnxruntime as ort
import numpy as np
import os
from tokenizers import Tokenizer

# Configuration: Paths to the AI models (Relative to waf_core folder)
MODEL_PATH = "../ml_engine/models/neurowall.onnx"
TOKENIZER_PATH = "../ml_engine/tokenizer/tokenizer.json"

class WAFEngine:
    def __init__(self):
        self.use_mock = False
        self.session = None
        self.tokenizer = None
        
        # 1. Try to load the Real AI (Member B's work)
        if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
            print("🧠 Loading Real AI Model (ONNX)...")
            try:
                self.session = ort.InferenceSession(MODEL_PATH)
                self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
                print("✅ AI Model Loaded Successfully.")
            except Exception as e:
                print(f"⚠️ Error loading AI: {e}. Switching to Mock Mode.")
                self.use_mock = True
        else:
            # 2. If files are missing, use Mock Mode (Safe for Dev)
            print("⚠️ Model files not found. Switching to Mock Mode.")
            self.use_mock = True

    def sigmoid(self, x):
        # Converts raw math numbers to a percentage (0.0 to 1.0)
        return 1 / (1 + np.exp(-x))

    def predict(self, text):
        """
        Analyzes text and returns: (is_malicious: bool, confidence_score: float, engine_name: str)
        """
        # --- MOCK MODE LOGIC (The "Digital Bouncer" simulation) ---
        if self.use_mock:
            text_lower = text.lower()
            # Simple keyword rules to simulate "Intelligence"
            if "select" in text_lower and "from" in text_lower:
                return True, 0.99, "Mock-Engine"
            if "<script>" in text_lower:
                return True, 0.98, "Mock-Engine"
            if "union" in text_lower and "select" in text_lower:
                return True, 0.95, "Mock-Engine"
            return False, 0.01, "Mock-Engine"

        # --- REAL AI LOGIC (Runs when .onnx file exists) ---
        try:
            # 1. Tokenize (Convert text to numbers)
            encoded = self.tokenizer.encode(text)
            
            # 2. Pad/Truncate to 128 tokens (Standard BERT size)
            max_len = 128
            input_ids = encoded.ids[:max_len] + [0] * (max_len - len(encoded.ids[:max_len]))
            attention_mask = encoded.attention_mask[:max_len] + [0] * (max_len - len(encoded.attention_mask[:max_len]))

            # 3. Run the AI Model (ONNX)
            inputs = {
                "input_ids": np.array([input_ids], dtype=np.int64),
                "attention_mask": np.array([attention_mask], dtype=np.int64)
            }
            # Run inference
            logits = self.session.run(None, inputs)[0]
            
            # 4. Calculate Confidence
            probs = self.sigmoid(logits)
            threat_score = float(probs[0][1]) # Index 1 is usually "Malicious"
            
            # 5. Decision Threshold (80%)
            return threat_score > 0.8, threat_score, "NeuroWall-AI"

        except Exception as e:
            print(f"Inference Error: {e}")
            # Fail Open: If AI crashes, let traffic through to avoid downtime
            return False, 0.0, "Error-FailOpen"

# Initialize the engine once (Singleton Pattern)
waf_engine = WAFEngine()