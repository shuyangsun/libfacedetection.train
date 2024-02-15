FROM nvcr.io/nvidia/tensorrt:23.11-py3

# Install NVIDIA container runtime following instructions at
# https://stackoverflow.com/a/61737404/2177724

# Copy "yolox_x.pth" is under the "model" directory in this repo first.

# Build image:
# docker image build . -t ssml/yolox:latest

# Run image:
# docker run --gpus device=0 ssml/yolox:latest

ENV CUDA_HOME=/usr/local/cuda
ENV BATCH 32

RUN  echo "    IdentityFile ~/.ssh/id_rsa" >> /etc/ssh/ssh_config

# lifacedetection.train
RUN mkdir /libfacedetection.train
WORKDIR /libfacedetection.train

RUN apt update
RUN apt-get install python3.10-venv ffmpeg libsm6 libxext6 -y
RUN python -m venv .venv
RUN source .venv/bin/activate

ADD requirements_exact.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install mmcv-full==1.3.17


WORKDIR /libfacedetection.train
ADD ./ /libfacedetection.train


# Would need to enable ssh config to pull from github directly. Work around is to clone the repo's locally and switch to right branch

# Install mmdeploy
# pip install -e git+https://github.com/open-mmlab/mmdeploy.git@c73756366e374507b5d619cfe06deea72d2ad61d#egg=mmdeploy
# git clone https://github.com/open-mmlab/mmdeploy.git
# git checkout c73756366e374507b5d619cfe06deea72d2ad61d
WORKDIR /libfacedetection.train/mmdeploy
RUN pip install -e .


# Install mmdet
#pip install -e git+ssh://git@github.com/shuyangsun/libfacedetection.train.git@f2a84b21ddc28bec2ee9afd1eab9d80d60798461#egg=mmdet
#git clone git@github.com:shuyangsun/libfacedetection.train.git
#git checkout f2a84b21ddc28bec2ee9afd1eab9d80d60798461
WORKDIR /libfacedetection.train/libfacedetection.train
RUN pip install -e .


#pip install git+ssh://git@github.com/shuyangsun/torch2trt.git@f92fc974f47abe6294f02e6ec147f7e617a48755#egg=torch2trt
#git clone git@github.com:shuyangsun/torch2trt.git
#git checkout f92fc974f47abe6294f02e6ec147f7e617a48755
WORKDIR /libfacedetection.train/torch2trt
RUN pip install -e .

WORKDIR /libfacedetection.train
ENTRYPOINT python tools/trt.py /libfacedetection.train/configs/yunet_n_deploy.py /libfacedetection.train/weights/yunet_n.pth  -i 1024 -b ${BATCH} -d cuda:0 --out ./

