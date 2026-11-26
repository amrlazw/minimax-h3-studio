FROM python:3.12-slim
RUN pip install --no-cache-dir torch torchao diffusers transformers accelerate av pillow
# Qwen3VLVideoProcessor needs torchvision; torchaudio resamples ref2va audio/video soundtracks to 32 kHz.
RUN pip install --no-cache-dir torchvision torchaudio
RUN pip install --no-cache-dir fastapi uvicorn python-multipart
COPY server.py index.html healthcheck.py /app/
COPY guides /app/guides/
WORKDIR /app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
