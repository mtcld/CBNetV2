from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import matplotlib.pyplot as plt
import cv2
import json
import os
from collections import defaultdict
import numpy as np
from skimage import draw
import pandas as pd
import copy
import time
from tqdm import tqdm

carpart_config = '../checkpoints/carpart/swa_carpart.py'
carpart_checkpoint = '../checkpoints/carpart/swa_carpart.pth'
carpart_model = init_detector(carpart_config, carpart_checkpoint, device='cuda:2')
classes = carpart_model.CLASSES

imgs_folder = '../data/coco_datasets/datasets/scratch/images/'
print(len([entry for entry in os.listdir(imgs_folder) if os.path.isfile(os.path.join(imgs_folder, entry))]))


# file_name = 'test_train_valid_combination'
json_path='../data/coco_datasets/datasets/scratch/annotations/test_train_valid_combination.json'
total_data = json.load(open(json_path))

def inference_result(model, image, img_id):
    
    result = inference_detector(model,image.copy())
    out_image,pred_boxes,pred_segms,pred_labels,pred_scores = show_result_pyplot(model,image.copy(),result,score_thr=0.7)

    carpart_inference_result = {}
    carpart_inference_result[img_id] = {}
    carpart_inference_result[img_id]['labels'] = pred_labels.tolist()
    carpart_inference_result[img_id]['bboxes'] = (np.array(pred_boxes).astype(np.uint8)).tolist()
    carpart_inference_result[img_id]['scores'] = pred_scores.tolist()
    
    segs = []
    for seg in pred_segms:
        contours, hierarchy = cv2.findContours(np.array(seg).astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            contour = contour.tolist()
            segs.append(contour)
    carpart_inference_result[img_id]['segmentations'] = segs
    
    return carpart_inference_result

carpart_inference_results = {}
file_name = 'carpart_inference_results'
json_path='../data/coco_datasets/datasets/scratch/annotations/'
inference_results_path = os.path.join(json_path, file_name+'.json')

with open(inference_results_path, 'w', encoding='utf-8') as json_file:
    json.dump(carpart_inference_results, json_file)


for img in tqdm(total_data['images']):
    img_name = img['file_name']
    img_id = img['id']
    image = cv2.imread(os.path.join(imgs_folder, img_name))
    
    carpart_inference_results.update(inference_result(carpart_model, image, img_id))

with open(inference_results_path, mode='w', encoding='utf-8') as json_file:
        json.dump(carpart_inference_results, json_file, ensure_ascii=False, indent=4)
    
