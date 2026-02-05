FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget curl git build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# Install CPU-only PyTorch and other requirements
# We remove specific torch versions from requirements.txt to prefer the CPU build installed below
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    sed -i '/torch==/d' requirements.txt && \
    sed -i '/torchvision==/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the CPU rasterizer extension
WORKDIR /app/decalib/utils/rasterizer
RUN python setup.py build_ext --inplace

WORKDIR /app
