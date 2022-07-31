
import argparse
import glob
import json
import os
import os.path as ops
import shutil

import cv2
import numpy as np


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='The origin path of unzipped tusimple dataset')
    return parser.parse_args()


def process_tusimple_dataset(src_dir):
    traing_folder_path = ops.join(ops.split(src_dir)[0], 'training')
    os.makedirs(traing_folder_path, exist_ok=True)

    gt_image_dir = ops.join(traing_folder_path, 'gt_image')
    gt_binary_dir = ops.join(traing_folder_path, 'gt_binary_image')
    gt_instance_dir = ops.join(traing_folder_path, 'gt_instance_image')

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)

    for idx, json_label_path in enumerate(glob.glob('{:s}/*.json'.format(src_dir))):
        get_image_to_folders(json_label_path, gt_image_dir, gt_binary_dir, gt_instance_dir, src_dir, idx)
    #gen_train_sample(src_dir, gt_binary_dir, gt_instance_dir, gt_image_dir)
    #split_train_txt(src_dir)

def get_image_to_folders(json_label_path, gt_image_dir, gt_binary_dir, gt_instance_dir, src_dir, idx):
    json_gt = [json.loads(line) for line in open(json_label_path)]
    for id, gt in enumerate(json_gt):
        filename = '{}_{}.png'.format(idx, id)
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']

        raw_img = cv2.imread(raw_file)
        
        gt_lanes_vis = [[(x,y) for (x,y) in zip(lane, y_samples) if x >=0] for lane in gt_lanes]
        img_vis = raw_img

        bin_background = np.zeros_like(raw_img)
        inst_background = np.zeros_like(raw_img)
        skip=False

        #for lane in gt_lanes_vis:
            

        colors = [[70,70,70], [120,120,120], [20,20,20], [170,170,170]]
        color_bw = [[255,255,255], [255,255,255], [255,255,255], [255,255,255]]
        for i in range(len(gt_lanes_vis)):
            if len(gt_lanes_vis[i]) == 0:
                skip=False
                break
            else: 
                inst_mask = cv2.polylines(inst_background, np.int32([gt_lanes_vis[i]]), isClosed=False, color=colors[i], thickness=5)
                bin_mask = cv2.polylines(bin_background, np.int32([gt_lanes_vis[i]]), isClosed=False, color=color_bw[i], thickness=5)
                skip=False

        if skip==True:
            print('Number of lanes is not 4!')
        else:
            shutil.copy(raw_file, ops.join(gt_image_dir, filename))
            cv2.imwrite(ops.join(gt_binary_dir, filename), bin_mask)
            cv2.imwrite(ops.join(gt_instance_dir, filename), inst_mask)


if __name__ == '__main__':
    args = init_args()
    print(os.getcwd())

    src = glob.glob(os.path.join(os.getcwd(), 'data\dataset\Town03'))
    print(src)
    # process_tusimple_dataset(args.src_dir)
    process_tusimple_dataset(src[0])

# json_label_path = './data/dataset/Town03/train_gt.json'
json_label_path = '\data\dataset\Town03\train_gt.json'