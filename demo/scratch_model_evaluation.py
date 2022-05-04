from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import json
from collections import defaultdict
from skimage import draw
import pandas as pd
from tqdm import tqdm

scratch_config = '/mmdetection/checkpoints/06_04_2022/scratch/scratch_merimen.py'
scratch_checkpoint = '/mmdetection/checkpoints/06_04_2022/scratch/cbn_clean_scratch_Merimen.pth'
scratch_model = init_detector(scratch_config, scratch_checkpoint, device='cuda:2')

imgs_dir = '/mmdetection/data/coco_datasets/datasets/testsets/scratch/images/'
gt_data = json.load(open('/mmdetection/data/coco_datasets/datasets/testsets/scratch/annotations/clean_test.json'))
imgToAnn = defaultdict(list)
for ann in gt_data['annotations']:
    imgToAnn[ann['image_id']].append(ann)

def compute_iou(ground_truth_mask, predict_mask):
    area1 = np.sum(ground_truth_mask)
    area2 = np.sum(predict_mask)
#     print('gt area: {}, pred area: {}'.format(area1, area2))
    #intersections
    intersection = np.sum(np.logical_and(ground_truth_mask, predict_mask))
#     print('intersection',intersection)
    union = np.sum(np.add(ground_truth_mask,predict_mask)>0)- intersection
#     print('union',union)
    iou = intersection/union
#     print(iou)
    return iou
def draw_binary_mask(seg, img_shape):
    seg = seg.reshape(-1,2).astype(np.int32)
    #print('debug : ',img_shape)
    mask = np.zeros(img_shape)
    #print(seg)
    mask = cv2.fillPoly(mask, [seg],1)
# #     polygon = np.array(seg)
#     mask = draw.polygon2mask(img_shape, polygon)
    return mask

def load_predicted_result(img, conf_score=0.01):
    result = inference_detector(scratch_model, img.copy())
    out_image,pd_boxes,pd_segs,pd_labels,pd_scores = show_result_pyplot(scratch_model,img.copy(),result,score_thr=conf_score)
    return pd_segs,pd_scores,out_image

def compute_eval_metrics(img_shape, pred_boxes,gt_boxes):
    '''
    return (tp, precision, recall, f1_score)
    '''
#     gt_labels = [0]* len(gt_boxes)
#     fp = 0
    tp = 0
    if pred_boxes.shape[0] == 0:
        return 0,0,0,0
    for pred_box in pred_boxes:
        ious = []
        for gt_box in gt_boxes:
            gt_box = draw_binary_mask(gt_box, img_shape)
            ious.append(compute_iou(gt_box, pred_box))
      
        if len(ious) == 0:
            return 0,0,0,0
        for idx, iou in enumerate(ious):
#             if gt_labels[idx] == -1:
#                 continue
            if iou >= 0.5:
                tp+=1
    if tp == 0:
        return 0,0,0,0
    if tp > pred_boxes.shape[0]:
        precision = 1
    else:
        precision = tp/pred_boxes.shape[0]
        
    if tp > len(gt_boxes):
        recall = 1
    else:
        recall = tp/len(gt_boxes)
    f1_score = 2*((precision*recall)/(precision+recall))

    return tp,precision,recall,f1_score

conf_scores = np.arange(0,1, 0.1)
pre_df = pd.DataFrame(columns = ['img_id', 'precision_0', 'precision_1', 'precision_2', 'precision_3', 'precision_4', 'precision_5', 'precision_6', 'precision_7', 'precision_8', 'precision_9'])
recall_df = pd.DataFrame(columns = ['img_id', 'recall_0','recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8', 'recall_9'])
f1_df = pd.DataFrame(columns = ['img_id', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'f1_4', 'f1_5', 'f1_6', 'f1_7', 'f1_8', 'f1_9'])
pre_df.to_csv('/mmdetection/demo/statistical_charts/precision.csv', index=False)
recall_df.to_csv('/mmdetection/demo/statistical_charts/recall.csv', index=False)
f1_df.to_csv('/mmdetection/demo/statistical_charts/f1_score.csv', index=False)

for i,img in enumerate(tqdm(gt_data['images'])):
    #load ground truth masks
    img_id = img['id']
    img_name = img['file_name']
    img = cv2.imread(os.path.join(imgs_dir, img_name))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape[:2]
    gt_anns = imgToAnn[img_id]
    gt_segs = [np.array(ann['segmentation']) for ann in gt_anns]
    precision_row = [img_id]
    recall_row = [img_id]
    f1_row = [img_id]
    #load predicted masks
    predicted_results = load_predicted_result(img)
    pred_segs = predicted_results[0]
    pred_scores = predicted_results[1]
    
    for conf_score in conf_scores:
        pd_segs = np.array([pred_seg for i,pred_seg in enumerate(pred_segs) if pred_scores[i]>=conf_score])
        tp, _precision, _recall, f1_score = compute_eval_metrics(img_shape, pd_segs,gt_segs)
        
        precision_row.append(_precision)
        recall_row.append(_recall)
        f1_row.append(f1_score)
        
    p_df = pd.DataFrame([precision_row])
    p_df.to_csv('/mmdetection/demo/statistical_charts/precision.csv', mode = 'a', index=False, header=False)
    
    r_df = pd.DataFrame([recall_row])
    r_df.to_csv('/mmdetection/demo/statistical_charts/recall.csv', mode = 'a', index=False, header=False)
    
    f_df = pd.DataFrame([f1_row])
    f_df.to_csv('/mmdetection/demo/statistical_charts/f1_score.csv', mode = 'a', index=False, header=False)
    