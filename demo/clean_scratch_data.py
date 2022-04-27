import json
import argparse
import pandas as pd
import copy
import numpy as np
import cv2

def compute_area(seg):
    seg = np.array(seg).reshape(-1,2).astype(np.int32)
    area = cv2.contourArea(seg)
    if area == 0:
        area = 1e-5
    return area

def double_check(data):
     img_ids = [img['id'] for img in data['images']]
     img_ids_in_annos = set([ann['image_id'] for ann in data['annotations']])
     differential_quant = len(img_ids) - len(img_ids_in_annos)
     if (differential_quant <= 0):
         print("check 'double_check' function again")
     else:
         print('have positive differential quantity')
         new_images = []
         for img in data['images']:
             if img['id'] in img_ids_in_annos:
                 new_images.append(img)
         data['images'] = new_images

     return data

def clean_data(input_path, output_path):
     data = json.load(open(input_path))   
     seg_area = {}
     seg_img_ratio = {}
     img_id = {}
     for ann in data['annotations']:
         seg_area[ann['id']] = compute_area(ann['segmentation'])
         img_id[ann['id']] = ann['image_id']
        
         for img in data['images']:
             if ann['image_id'] == img['id']:
                 img_area = img['width'] * img['height']
                 break
         seg_img_ratio[ann['id']] = seg_area[ann['id']]/img_area
     
     df = pd.DataFrame({'anno_id': seg_area.keys(),'img_id': img_id.values(),'seg_area': seg_area.values(),'seg_img_ratio': seg_img_ratio.values()}) 
    
     upper_limit = 0.15
     lower_limit = np.exp(-11)
     under_lower_ratio_outlier = df[(df.seg_img_ratio<=lower_limit)]
     above_upper_ratio_outlier = df[(df.seg_img_ratio>=upper_limit)]
    
     removed_ann_id = list(under_lower_ratio_outlier.anno_id)
     removed_img_id = list(set(above_upper_ratio_outlier.img_id))
     new_json = copy.deepcopy(data)

     for ann in new_json['annotations']:
         if ann['image_id'] in removed_img_id:
             removed_ann_id.append(ann['id'])

     new_json['images'] = [img for img in new_json['images'] if img['id'] not in removed_img_id]
     new_json['annotations'] = [ann for ann in new_json['annotations'] if ann['id'] not in removed_ann_id]
    #  print(len(new_json['images']), len(new_json['annotations']))
     new_json = double_check(new_json)
    
     with open(output_path, 'w', encoding='utf-8') as json_file:
         json.dump(new_json, json_file, ensure_ascii=False, indent=4)

def main():
     parser = argparse.ArgumentParser()
     parser.add_argument('input_path', type=str, help='path to input json file')
     parser.add_argument('output_path', type=str, help='path to output json file')
    
     args = parser.parse_args()
    
     input_path = args.input_path
     output_path = args.output_path
     clean_data(input_path, output_path)
if __name__ == '__main__':
     main()   
