FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY i2t ./i2t
COPY data/captions.txt data/captions_ro.txt ./data/
ENV I2T_STORAGE=/app/storage I2T_RESULTS=/app/results HF_HOME=/app/.hf
VOLUME ["/app/storage", "/app/.hf"]
EXPOSE 8000
CMD ["python", "-m", "i2t.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
