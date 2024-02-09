FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Pythonと必要なパッケージをインストール
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    sudo \
    vim \
    wget \
    git \
    libgl1-mesa-dev \
    libglib2.0-0 \
    curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/local/src/* /tmp/*

# 不要なシンボリックリンクの設定ステップを削除し、必要なシンボリックリンクを設定
RUN rm -f /usr/bin/python && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python3

# その他の必要なパッケージをインストール
RUN pip install \
    openmim \
    gradio \
    transformers \
    addict \
    yapf \
    numpy \
    opencv-python \
    supervision==0.18.0 \
    ftfy \
    regex \
    pot \
    sentencepiece \
    tokenizers

# 特定のtorchと関連ライブラリをインストール
RUN pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0

# 特定のmmyoloとmmdetをインストール
RUN pip install mmyolo==0.6.0 mmdet==3.0.0

# yolo-worldをインストール
RUN git clone https://github.com/hiroking0523/YOLO-World.git /YOLO-World
RUN cd /YOLO-World && pip install -e .