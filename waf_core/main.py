import time
import re
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from transformers import RobertaTokenizerFast
import json
import os
from datetime import datetime, timedelta

app = FastAPI()

class SystemState:
    mode = "hybrid"

metrics = {"total": 0, "blocked": 0, "regex_blocks": 0, "ai_blocks": 0, "path_blocks": 0, "latency": 0.0}
latencies = []
recent_logs = []

class ModeUpdate(BaseModel):
    mode: str

REGEX_RULES = [
    (re.compile(r"(?i)(union\s+select|select\s+\*|drop\s+table|<script>|javascript:)"), "regex_blocks", "Regex (SQLi/XSS)"),
    (re.compile(r"(?i)(\.\./\.\./etc/passwd)"), "path_blocks", "Regex (Path Traversal)"),
]

try:
    session = ort.InferenceSession(os.path.join("./model_store", "neurowall.onnx"))
    tokenizer = RobertaTokenizerFast.from_pretrained("./model_store", max_len=128)
except:
    pass

def update_latency(new_lat):
    latencies.append(new_lat)
    if len(latencies) > 20: latencies.pop(0)
    metrics["latency"] = sum(latencies) / len(latencies)

def add_log(route, payload, verdict, engine):
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    log_entry = {
        "Time": ist_time.strftime("%I:%M:%S %p"),
        "Route": route,
        "Payload": payload[:60] + "..." if len(payload) > 60 else payload,
        "Verdict": verdict,
        "Engine": engine
    }
    recent_logs.insert(0, log_entry)
    if len(recent_logs) > 8:
        recent_logs.pop()

def formatted_response(data):
    return Response(content=json.dumps(data) + "\n", media_type="application/json")

@app.post("/set_mode")
def set_mode(update: ModeUpdate):
    SystemState.mode = update.mode
    return formatted_response({"status": "success"})

@app.get("/metrics")
def get_metrics():
    return {"metrics": metrics, "logs": recent_logs}

@app.post("/analyze")
async def inspect_traffic(request: Request):
    metrics["total"] += 1
    body = await request.json()
    payload = body.get("payload", "")
    route = body.get("route", "")
    start_time = time.time()

    if route.endswith(('.css', '.js', '.png', '.jpg')):
        return formatted_response({"verdict": "ALLOW"})

    # Lock the mode for this exact request
    current_mode = SystemState.mode 

    # 1. iGPU / REGEX ONLY (STRICT LOCK)
    if current_mode == "regex_only":
        for rule, metric_key, engine_name in REGEX_RULES:
            if rule.search(payload):
                metrics["blocked"] += 1
                metrics[metric_key] += 1
                update_latency((time.time() - start_time) * 1000)
                add_log(route, payload, "🚫 BLOCKED", engine_name)
                return formatted_response({"verdict": "BLOCK"})
        
        # Exits immediately. AI is bypassed.
        update_latency((time.time() - start_time) * 1000)
        add_log(route, payload, "✅ ALLOWED", "Regex Passed")
        return formatted_response({"verdict": "ALLOW"})

    # 2. dGPU / AI ONLY
    elif current_mode == "ai_only":
        inputs = tokenizer(payload, padding="max_length", truncation=True, max_length=128, return_tensors="np")
        onnx_inputs = {session.get_inputs()[0].name: inputs['input_ids'].astype(np.int64), session.get_inputs()[1].name: inputs['attention_mask'].astype(np.int64)}
        logits = session.run(None, onnx_inputs)[0]
        prob = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        update_latency((time.time() - start_time) * 1000)
        
        if prob[0][1] > 0.80:
            metrics["blocked"] += 1
            metrics["ai_blocks"] += 1
            add_log(route, payload, "🚫 BLOCKED", "NeuroWall")
            return formatted_response({"verdict": "BLOCK"})
            
        add_log(route, payload, "✅ ALLOWED", "NeuroWall Clean")
        return formatted_response({"verdict": "ALLOW"})

    # 3. HYBRID (Mux Switch)
    else:
        for rule, metric_key, engine_name in REGEX_RULES:
            if rule.search(payload):
                metrics["blocked"] += 1
                metrics[metric_key] += 1
                update_latency((time.time() - start_time) * 1000)
                add_log(route, payload, "🚫 BLOCKED", engine_name)
                return formatted_response({"verdict": "BLOCK"})

        inputs = tokenizer(payload, padding="max_length", truncation=True, max_length=128, return_tensors="np")
        onnx_inputs = {session.get_inputs()[0].name: inputs['input_ids'].astype(np.int64), session.get_inputs()[1].name: inputs['attention_mask'].astype(np.int64)}
        logits = session.run(None, onnx_inputs)[0]
        prob = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        update_latency((time.time() - start_time) * 1000)

        if prob[0][1] > 0.80:
            metrics["blocked"] += 1
            metrics["ai_blocks"] += 1
            add_log(route, payload, "🚫 BLOCKED", "NeuroWall")
            return formatted_response({"verdict": "BLOCK"})
        
        add_log(route, payload, "✅ ALLOWED", "NeuroWall Clean")
        return formatted_response({"verdict": "ALLOW"})