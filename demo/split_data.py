import argparse
import json
import copy
import random
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('orgin_path',type=str, help='path to original data source')
    parser.add_argument('des_path', type=str,help='destination path of extracted data')
    parser.add_argument('des_file_name', type=str, help='destination file name')
    parser.add_argument('num_imgs', type=int, help = 'number of images you want to extract')

    args = parser.parse_args()
    return args
def imgs_annos_mapping(data, imgs):
    img_ids = [img['id'] for img in imgs]
    dest_anns = [ann for ann in data['annotations'] if ann['image_id'] in img_ids]
    return dest_anns

def main():
    args = parse_args()
    origin_data = json.load(open(args.orgin_path))
    # extracted_imgs = random.sample(origin_data['images'], args.num_imgs)
    extracted_imgs = origin_data['images'][:args.num_imgs]
    extracted_anns = imgs_annos_mapping(origin_data, extracted_imgs)

    des_data = copy.deepcopy(origin_data)
    des_data['images'] = extracted_imgs
    des_data['annotations'] = extracted_anns
    file_name = args.des_file_name + '.json'
    des_path = os.path.join(args.des_path, file_name)

    with open(des_path, 'w', encoding = 'utf-8') as json_file:
        json.dump(des_data, json_file, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    main()