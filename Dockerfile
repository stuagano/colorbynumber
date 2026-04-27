FROM python:3.11-slim

WORKDIR /app

# Install system deps for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps with exact pins for ALL critical packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        starlette==0.36.3 \
        fastapi==0.109.2 \
        jinja2==3.1.2 \
        huggingface_hub==0.24.7 \
    && pip freeze > /app/frozen-deps.txt

# Copy application code
COPY . .

EXPOSE 10000

CMD ["python", "app.py"]
