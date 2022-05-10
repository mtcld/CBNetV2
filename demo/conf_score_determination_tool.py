import argparse
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
import random
from hyperopt import hp, fmin, tpe, Trials
import pickle

# def objective_func(space, gt_data, imgs_dir, model):
#     conf_score = space['conf_score']['conf_score']
#     print('debug : ',conf_score,' space : ',space)
#     precision_mean, recall_mean, f1_mean = compute_mean_metrics(gt_data, conf_score, imgs_dir, model)
#     return -f1_mean

def compute_iou(ground_truth_mask, predict_mask):
    area1 = np.sum(ground_truth_mask)
    area2 = np.sum(predict_mask)
    intersection = np.sum(np.logical_and(ground_truth_mask, predict_mask))
    iou = intersection/(area1+area2-intersection)
    return iou

def draw_binary_mask(seg, img_shape):
    seg = seg.reshape(-1,2).astype(np.int32)
    mask = np.zeros(img_shape)
    mask = cv2.fillPoly(mask, [seg],1)
    return mask
def img_to_ann(gt_data):
    imgToAnn = defaultdict(list)
    for ann in gt_data['annotations']:
        imgToAnn[ann['image_id']].append(ann)
    return imgToAnn

def load_predicted_result(model, img, conf_score=0.001):
    result = inference_detector(model, img.copy())
    out_image,pd_boxes,pd_segs,pd_labels,pd_scores = show_result_pyplot(model,img.copy(),result,score_thr=conf_score)
    return pd_segs,pd_scores,out_image
def compute_eval_metrics(img_shape, pred_boxes,gt_boxes):
    '''
    return (tp,fp,fn, precision, recall, f1_score)
    '''
    gt_labels = [0]* len(gt_boxes)
    fp = 0
    tp = 0
    if pred_boxes.shape[0] == 0:
        return 0,0,0,0,0,0
    
    for pred_box in pred_boxes:
        #print(gt_labels)
        ious = []
        for gt_box in gt_boxes:
            gt_box = draw_binary_mask(gt_box, img_shape)
            ious.append(compute_iou(gt_box, pred_box))
      
        if len(ious) == 0:
            return 0,0,0,0,0,0
        #print('ious : ',ious)
        idx = np.argmax(ious)

        if gt_labels[idx] == -1 and ious[idx] >= 0.25:
            continue

        if ious[idx] >= 0.25 :
            gt_labels[idx] = -1
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - tp
    if tp == 0:
        precision  = 0
        recall = 0
        f1_score = 0
    else:
        precision = tp/(tp+fp)
        recall = tp/(len(gt_boxes))
        f1_score = 2*((precision*recall)/(precision+recall))

    return tp,fp,fn,precision,recall,f1_score

def compute_mean_metrics(gt_data, conf_score,imgs_dir, model):
    precision = 0
    recall = 0
    f1 = 0
    num_random_images = 1000
    imgs = random.sample(gt_data['images'], num_random_images)
    imgToAnn = img_to_ann(gt_data)
    for img in tqdm(imgs):
        #load ground truth masks
        img_id = img['id']
        img_name = img['file_name']
        img = cv2.imread(os.path.join(imgs_dir, img_name))
        img_shape = img.shape[:2]
        gt_anns = imgToAnn[img_id]
        gt_segs = [np.array(ann['segmentation']) for ann in gt_anns]
  
        #load predicted masks
        predicted_results = load_predicted_result(model, img)
        pred_segs = predicted_results[0]
        
        pred_scores = predicted_results[1]
        #print('debug 2:',len(pred_scores),len(pred_segs),conf_score)
        
        pd_segs = np.array([pred_seg for pred_seg,pred_score in zip(pred_segs,pred_scores) if pred_score>=conf_score])
        tp,fp,fn, _precision, _recall, f1_score = compute_eval_metrics(img_shape, pd_segs,gt_segs)
        precision += _precision
        recall += _recall
        f1 += f1_score
    precision_mean = precision/num_random_images
    recall_mean = recall/num_random_images
    f1_mean = f1/num_random_images
    return precision_mean, recall_mean, f1_mean

      
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_config', type = str, help='path to model config file')
    parser.add_argument('model_checkpoint', type=str, help='path to model checkpoint file')
    parser.add_argument('images_path', type=str, help='path to images folder')
    parser.add_argument('json_file', type=str, help='path to annotating json file')

    args = parser.parse_args()
    return args
    
def main(args):
    model = init_detector(args.model_config, args.model_checkpoint, device = 'cuda:1')
    imgs_dir = args.images_path
    gt_path = args.json_file
    gt_data = json.load(open(gt_path))

    conf_f1_df = pd.DataFrame(columns= ['conf_score', 'f1_score'])
    conf_f1_df.to_csv('/mmdetection/demo/statistical_charts/conf_f1_searching_pair.csv', index=False)
    def objective_func(conf_score):
        #conf_score = conf_score['conf_score']
        precision_mean, recall_mean, f1_mean = compute_mean_metrics(gt_data, conf_score, imgs_dir, model)
        row = [conf_score, f1_mean]
        new_row_df = pd.DataFrame([row])
        new_row_df.to_csv('/mmdetection/demo/statistical_charts/conf_f1_searching_pair.csv', mode= 'a', index=False, header=False)
        # print(len(conf_f1_df))
        return -f1_mean

    conf_score_space = hp.uniform('conf_score', 0.001,1)
    #objective = objective_func(conf_score)

    trials = Trials()
    # trials = pickle.load(open("/mmdetection/demo/statistical_charts/conf_score_trials.p", "rb"))

    best = fmin(fn = objective_func, space = conf_score_space, algo = tpe.suggest,trials=trials, max_evals = 100)
    pickle.dump(trials, open("/mmdetection/demo/statistical_charts/conf_score_trials.p", "wb"))
#https://github.com/hyperopt/hyperopt/issues/267
#http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/
    return best, trials

if __name__ == '__main__':
    args = parse_args()
    best, trials = main(args)
    print('final conf result: ', best)
    print('best f1 score: ', -min(trials.losses()))

    # print('a list of dictionaries representing everything about the search',trials.trials)
    # print('a list of dictionaries returned by "objective" during the search ',trials.results)
    # print('a list of losses (float for each "ok" trial) ',trials.losses())
    # print(' a list of status strings ',trials.statuses())

    # conf_f1_df = pd.read_csv('/mmdetection/demo/statistical_charts/conf_f1_searching_pair.csv')
    # # conf_f1_df['f1_score'] = pd.to_numeric(conf_f1_df['f1_score'])
    # max_f1_index = conf_f1_df['f1_score'].idxmax()
    # print(conf_f1_df.loc[max_f1_index])