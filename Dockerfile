# espnet/Dockerfile for training
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential cmake \
    sox ffmpeg flac libsndfile1 locales \
 && rm -rf /var/lib/apt/lists/*

# UTF-8
RUN locale-gen en_US.UTF-8 zh_TW.UTF-8
ENV LANG=zh_TW.UTF-8 LC_ALL=zh_TW.UTF-8

# Python 部份
RUN python -m pip install --upgrade pip wheel setuptools
COPY requirements.docker.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ====== 複製需要的資料夾(相對位置要一樣) ======
WORKDIR /espnet
COPY espnet2/             /espnet/espnet2/
COPY egs2/yesno/asr1/     /espnet/egs2/yesno/asr1/
COPY egs2/TEMPLATE/       /espnet/egs2/TEMPLATE/
COPY tools/               /espnet/tools/
COPY espnet/              /espnet/espnet/
COPY utils/              /espnet/utils/

# 非 root 使用者，避免輸出成 root 權限
ARG USERNAME=espnet UID=1000 GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} && \
    chown -R ${USERNAME}:${USERNAME} /espnet
USER ${USERNAME}

# 讓 python 能直接找到 /espnet 下的模組（espnet2）
ENV PYTHONPATH=/espnet:$PYTHONPATH

# 預設進到 asr1
WORKDIR /espnet/egs2/yesno/asr1
CMD ["/bin/bash"]

#=======================================================================

# # espnet/Dockerfile for inference
# FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# ARG DEBIAN_FRONTEND=noninteractive
# ENV TZ=Asia/Taipei PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

# # Install dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential cmake python3-dev pkg-config \
#     libeigen3-dev libsndfile1 libsndfile1-dev libfftw3-dev \
#     protobuf-compiler libprotobuf-dev \
#     sox ffmpeg flac locales \
#  && rm -rf /var/lib/apt/lists/*

# # UTF-8 locale
# RUN python -m pip install --upgrade pip setuptools wheel \
#  && pip install "numpy<2" cython pybind11 scikit-build

# RUN apt-get update && apt-get install -y --no-install-recommends python3-sentencepiece && rm -rf /var/lib/apt/lists/*

# # ===== Install ESPnet =====
# WORKDIR /espnet
# COPY requirements.txt /tmp/requirements.txt
# RUN pip install -r /tmp/requirements.txt

# # ===== Inference script =====
# COPY pre_inference.sh /espnet/inference.sh
# RUN chmod +x /espnet/inference.sh

# CMD ["/espnet/inference.sh"]