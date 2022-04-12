import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import json
from collections import defaultdict, OrderedDict
from skimage import draw
from tqdm import tqdm


scratch_data_dir = '../data/coco_datasets/datasets/merimen_coco/19_02_2022'
dataType = 'total'
scratch_ann_file = '{}/scratch/annotations/{}.json'.format(scratch_data_dir,dataType)

scratch_mask_data = json.load(open(scratch_ann_file))

car_mask_path = '../data/coco_datasets/datasets/merimen_coco/19_02_2022/scratch/annotations/total_car_v2.json'
car_mask_data = json.load(open(car_mask_path))

img_to_scratch_anns = defaultdict(list)
for ann in scratch_mask_data['annotations']:
    img_to_scratch_anns[ann['image_id']].append(ann)

img_to_car_anns = defaultdict(list)
for ann in car_mask_data['annotations']:
    img_to_car_anns[ann['image_id']].append(ann)

print('Number of images without car mask annotation: ', len(car_mask_data['images'])-len(img_to_car_anns))

def compute_area(seg):
    seg = np.array(seg).reshape(-1,2).astype(np.int32)
    area = cv2.contourArea(seg)
    if area == 0:
        area = 1e-5
    return area

def draw_binary_mask(seg, img_shape):
    seg = np.array(seg).reshape(-1,2)
    polygon = np.array(seg)
    mask = draw.polygon2mask(img_shape,polygon)
    return mask
def intersection_check(original_image, cnt1, cnt2):
    img_shape = original_image.shape[:2]
    mask1 = draw_binary_mask(cnt1, img_shape)
    mask2 = draw_binary_mask(cnt2, img_shape)
    intersection = np.logical_and(mask1, mask2)
    return intersection.any()

carmask_area = OrderedDict()
carmask_id = {}
scratch_car_ratio = {}

for img in tqdm(car_mask_data['images'][:10]):
    img_name = img['file_name']
    image = io.imread('%s/scratch/images/%s'%(scratch_data_dir,img_name))
    img_id = img['id']

    scratch_anns = img_to_scratch_anns[img_id]
    car_anns = img_to_car_anns[img_id]
    
    if len(car_anns) == 0:
        car_anns = [{'id': img_id, 'segmentation': [0,0, img['width'],0, img['width'],img['height'], 0, img['height']]}]
    for scratch_ann in scratch_anns:
        scratch_ann_id = scratch_ann['id']
        scratch_seg = scratch_ann['segmentation']
        scratch_area = compute_area(scratch_seg)

        for car_ann in car_anns:
            car_ann_id = car_ann['id']
            car_seg = car_ann['segmentation']

            if intersection_check(image, scratch_seg, car_seg):
                car_area = compute_area(car_seg)                
            else:
                car_area = img['width']*img['height']
            ratio = scratch_area/car_area

            carmask_area[scratch_ann_id] = car_area
            carmask_id[scratch_ann_id] = car_ann_id
            scratch_car_ratio[scratch_ann_id] = ratio

print('debug : ',carmask_area)
print(carmask_area.values())
#print(len(carmask_area), len(carmask_id), len(scratch_car_ratio))

# feature_table = pd.read_csv('statistical_charts/merimen_scratch_report.csv')
# feature_table['carmask_area'] = carmask_area.values()
# feature_table['carmask_id'] = carmask_id.values()
# feature_table['scratch_car_ratio'] = scratch_car_ratio.values()

#feature_table.to_csv('statistical_charts/merimen_scratch_report.csv', index=False)

#print(feature_table.shape)