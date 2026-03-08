# /mnt/c/Neuro-Wall/waf_core/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install web server and AI tools
RUN pip install --no-cache-dir fastapi uvicorn onnxruntime transformers tokenizers
# Install the lightweight CPU-only PyTorch so it doesn't take 2GB of space
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy everything from waf_core into the container
COPY . /app/

# Open port 8000 for traffic
EXPOSE 8000

# Command to start the WAF
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]