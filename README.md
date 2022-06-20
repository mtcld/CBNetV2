# 0.Prepare data 
```
cd /data/motionscloud/training/carpart/data
python task/download.py  
python task/convert_dataset.py

```

# 1.Install training enviroment
```
sudo nvidia-docker run -itd --name car-train -p 4567:4567 -v /data/motionscloud:/data/motionscloud car-dat:latest
sudo docker exec -it *** /bin/bash 


cd /mmcv 
MMCV_WITH_OPS=1 pip install -e .

cd /data/motionscloud/training/carpart/CBNetV2
pip install -e .

pip uninstall pycocotools
pip install pycocotools
pip install mmpycocotools

```

# 2.Training model mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco

2.1: download pretrain 
```
cd /data/motionscloud/training/carpart/
wget https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip
unzip mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip 
```

2.3: train 

```
tools/dist_train.sh configs/cbnet/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py 2
```
