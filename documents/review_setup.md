# Download repo
- Clone repo  
`git clone https://github.com/thangnx183/CBNetV2`
- chekout review branch  
`git checkout feat/AIT-14-review-cbnetv2`

# Docker setup
- docker setup image  
`docker build -t cbnet_image docker/`
- setup container  
`nvidia-docker run --name cbn --gpus all --shm-size=32g -p 7000:7000 -p 7001:7001 -it -v /data/thang/CBNetV2/:/mmdetection -v /data/thang/mmdetection/data/:/mmdetection/data  cbnet_image`

# Install Apex (for mixed precision training only)
- download repo  
`git clone https://github.com/NVIDIA/apex`
- inside repo  
```
    cd apex
    git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
    python setup.py install --cuda_ext --cpp_ext
```

# Download pretrain [model](https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip)
# Sample of [config file](configs/cbnet/loose.py) 

