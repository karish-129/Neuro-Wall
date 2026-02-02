import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from engine import waf_engine  # Import the brain we just wrote

app = FastAPI(title="NeuroWall WAF Node")

# --- MIDDLEWARE: The Security Checkpoint ---
@app.middleware("http")
async def waf_middleware(request: Request, call_next):
    start_time = time.time()
    
    # 1. Whitelist: Don't block the documentation or Favicon
    if request.url.path in ["/docs", "/openapi.json", "/favicon.ico"]:
        return await call_next(request)

    # 2. Intercept Data: Read the user's input (Body & URL)
    try:
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8", errors="ignore")
        # Combine parameters and body for a full scan
        full_payload = f"{request.url.query_params} {body_text}"
    except:
        full_payload = ""

    # 3. The Interrogation: Ask the Engine
    try:
        is_attack, confidence, source = waf_engine.predict(full_payload)
    except Exception:
        # Fail-Safe: If engine crashes, default to Safe
        is_attack = False
        confidence = 0.0
        source = "Error"

    # 4. The Verdict
    if is_attack:
        latency = (time.time() - start_time) * 1000
        print(f"🚨 BLOCKED: {source} ({confidence:.2f}) - {latency:.2f}ms")
        
        # STOP! Return 403 Forbidden immediately.
        return JSONResponse(
            status_code=403,
            content={
                "status": "BLOCKED",
                "reason": "Malicious Payload Detected",
                "engine": source,
                "confidence": confidence,
                "latency": f"{latency:.2f}ms"
            }
        )

    # 5. Pass Through: Forward to the destination
    response = await call_next(request)
    
    # Add a custom header to prove it was scanned
    response.headers["X-Scanned-By"] = "NeuroWall-WAF"
    return response


# --- DUMMY APPLICATION (The Backend being protected) ---

@app.get("/")
def home():
    return {"message": "Welcome to the Secure Bank API"}

@app.post("/login")
def login(data: dict):
    return {"message": "Login Successful (WAF Allowed This)"}

@app.get("/search")
def search(q: str):
    return {"results": f"Searching for: {q}"}