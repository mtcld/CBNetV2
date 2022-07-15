import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import argparse

def create_box_df(damage_data):
    value_list = []
    for ann in damage_data['annotations']:
        for img in damage_data['images']:
            if img['id'] == ann['image_id']:
                img_w = img['height']
                img_h = img['width']
    
        value = (ann['id'], ann['image_id'], img_w, img_h, ann['bbox'][2], ann['bbox'][3], ann['bbox'][0], ann['bbox'][1])
        value_list.append(value)
        
    column_names = ['ann_id', 'img_id', 'img_width', 'img_height', 'box_width', 'box_height', 'xmin', 'ymin']
    box_df = pd.DataFrame(value_list, columns = column_names)
    box_df['new_img_w'], box_df['new_img_h'] = np.vectorize(_compute_new_static_size)(box_df['img_width'], box_df['img_height'], min_dimension, max_dimension)
    box_df['new_b_w'] = box_df['new_img_w']*box_df['box_width']/box_df['img_width']
    box_df['new_b_h'] = box_df['new_img_h']*box_df['box_height']/box_df['img_height']
    box_df['new_box_ar'] = box_df['new_b_h']/box_df['new_b_w']
    box_df['new_box_area'] = box_df['new_b_w']*box_df['new_b_h']

    # clean aspect ratio outliers
    box_ar_filtered = box_df[box_df['new_box_ar']<12]
    # clean area outliers
    box_ar_scale_filtered = box_ar_filtered[box_ar_filtered['new_box_area']<250000]

    return box_ar_scale_filtered
  
def cluster_ratio_area(box_df, output_dir):
    # cluster 5 ratios
    AR = (box_df['new_box_ar'].values).reshape(-1,1)

    ar_K = KMeans(5, random_state = 1)
    ar_labels = ar_K.fit(AR)
    ar_centers = list(np.array(ar_labels.cluster_centers_).reshape(1,-1)[0])

    # cluster 15 areas
    S = (box_ar_scale_filtered['new_box_area'].values).reshape(-1,1)
    area_K = KMeans(15, random_state = 1)
    area_labels = area_K.fit(S)
    area_centers = np.array(larea_abels.cluster_centers_)
    area_centers = np.array([i for i in area_centers])
    area_centers = np.sort(np.sqrt(area_centers), axis = None).reshape(5,-1)
    base_sizes = [round(c[0]) for c in area_centers]

    ele2 = [row[1] for row in area_15_ratio]
    ele3 = [row[2] for row in area_15_ratio]
    avg2=sum(ele2)/len(ele2)
    special_ele2 = [ele for ele in ele2 if ele>avg2]

    avg3 = sum(ele3)/len(ele3)
    special_ele3 = [ele for ele in ele3 if ele>avg3]
    picked_ele2 = [ele for ele in ele2 if ele not in special_ele2]
    picked_ele3 = [ele for ele in ele3 if ele not in special_ele3]
    picked_avg2 = sum(picked_ele2)/len(picked_ele2)
    picked_avg3 = sum(picked_ele3)/len(picked_ele3)
    picked_scales = [1., round(picked_avg2,3), round(picked_avg3,3)]

    anchor_settings = {'ratios': ar_centers, 'base_sizes': base_sizes, 'scales': picked_scales, 'areas': area_centers.tolist()}
    json_path = os.path.join(output_dir, 'anchor_settings_result.json')
    with open(json_path, 'w') as json_file:
        json.dump(anchor_settings, json_file, ensure_ascii=False, indent=4)
    


# function from Tensorflow Object Detection API to resize image
def _compute_new_static_size(width, height,min_dimension = 480,max_dimension = 1024):
    orig_height = height
    orig_width = width
    orig_min_dim = min(orig_height, orig_width)
  
    large_scale_factor = min_dimension / float(orig_min_dim)
    large_height = int(round(orig_height * large_scale_factor))
    large_width = int(round(orig_width * large_scale_factor))
    large_size = [large_height, large_width]
    if max_dimension:

        orig_max_dim = max(orig_height, orig_width)
        small_scale_factor = max_dimension / float(orig_max_dim)
        small_height = int(round(orig_height * small_scale_factor))
        small_width = int(round(orig_width * small_scale_factor))
        small_size = [small_height, small_width]
        new_size = large_size
    if max(large_size) > max_dimension:
        new_size = small_size
    else:
        new_size = large_size
    
    return new_size[1], new_size[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('damage_data', type=str, help='path to damage data')
    parser.add_argument('output_dir', type=str, help='output path to save anchor settings result')
    args = parser.parse_args()
    
    damage_data = json.load(open(args.damage_data))
    box_df = create_box_df(damage_data)
    cluster_ratio_area(box_df, args.output_dir)

if __name__ == '__main__':
    main()