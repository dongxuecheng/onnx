FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Update pip and set Tsinghua mirror
RUN python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# Install required packages
RUN pip3 install onnxruntime-gpu opencv-python "fastapi[standard]" PyYaml

RUN apt-get update && \
    apt-get install -y libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

