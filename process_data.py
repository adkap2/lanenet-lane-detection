import os
from telnetlib import PRAGMA_HEARTBEAT
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import glob
import matplotlib.pyplot as plt
import random


def read_dataset(file_path, num_samples=4):
    # Function:
    #     Read a file consisting of rows of training dataset path and
    #     seperate them into 3 variables (img_path, bin_path, inst_path)
    # Input:
    #     file_path   : text file consisting of training dataset path
    #     num_samples : number of dataset samples that you want to get from the file 
    # Output:
    #     img_path    : training RGB image paths
    #     bin_path    : training binary segmentation image paths
    #     inst_path   : training instance segmentation image paths

    img_path = []
    bin_path = []
    inst_path = []

    # assert exists(file_path), "Training file (txt) does not exist, please make sure the input file is right."

    num_files = sum(1 for line in open(file_path)) #number of records in file (excludes header)
    if num_files < num_samples:
        print('Number of samples is higher than the number of existing files. number of samples is setted to the number of existing files: ', num_files)
        num_samples = num_files
    skip_files = sorted(random.sample(range(1,num_files+1),(num_files-num_samples))) #the 0-indexed header will not be included in the skip list
    text_file = pd.read_csv(file_path, header=None, delim_whitespace=True, skiprows=skip_files, names=['img', 'bin', 'inst'])

    img_path = text_file['img'].values
    bin_path = text_file['bin'].values
    inst_path = text_file['inst'].values

    return img_path, bin_path, inst_path

def preprocessing(img_path, bin_path, inst_path, resize=(1280,720)):
    image_ds = []
    print("Generating image dataset...")
    for i, image_name in enumerate(img_path):
        image = cv2.imread(image_name)
        image = Image.fromarray(image)
        image = image.resize(resize)
        image = np.array(image, dtype=np.float32)
        image_ds.append(image)
        #break
        # if i == 100:
        #     break


     
    mask_ds = []
    print("Generating Masks")
    for i, image_name in enumerate(bin_path):

        image = cv2.imread(image_name, 0)
        im_gray = cv2.resize(image, resize)
        image = np.array(image, dtype=np.uint8)


        mask_ds.append(image)
        #break
        # if i == 100:
        #     break


    inst_ds = []
    print("Generating Instance")
    for i, image_name in enumerate(inst_path):

        image = cv2.imread(image_name, 0)
        image = cv2.resize(image, resize)
        image = np.array(image, dtype=np.uint8)
        inst_ds.append(image)
        #break
        # if i == 100:
        #     break


    
    print("Preprocessing done")
    return image_ds, mask_ds, inst_ds


def save_data(image_ds, mask_ds, inst_ds, save_path):
    assert len(image_ds) == len(mask_ds) == len(inst_ds)
    for i, image in enumerate(image_ds):
        cv2.imwrite(os.path.join(save_path, 'gt_image', 'image_{}.png'.format(i)), image)
        cv2.imwrite(os.path.join(save_path, 'gt_binary_image', 'mask_{}.png'.format(i)), mask_ds[i])

        cv2.imwrite(os.path.join(save_path, 'gt_instance_image', 'inst_{}.png'.format(i)), inst_ds[i])
    print('save data done')

if __name__=="__main__":

    root = os.getcwd()
    print(root)
    # img_path, bin_path, inst_path = read_txt(root, 'train')
    lst_images = glob.glob(os.path.join(root, 'gt_image', '*.png'))
    lst_masks = glob.glob(os.path.join(root, 'gt_binary_image', '*.png'))
    lst_inst = glob.glob(os.path.join(root, 'gt_instance_image', '*.png'))

    image_ds, mask_ds, inst_ds = preprocessing(lst_images, lst_masks, lst_inst)
    
    save_path = os.path.join(root, 'resized')

    save_data(image_ds, mask_ds, inst_ds, save_path)

    # read one image in 
    img_path = os.path.join(root, 'resized',  'gt_binary_image', 'mask_0.png')
    img = Image.open(img_path)
    # img = Image.fromarray(img)
    # img = plt.imread(img_path)
    #img = img[:,:,0]
    

    print(img.size)
    print(img.mode)
    #print(img.getbands())
    print(np.array(img).shape)
    print(np.array(img))
    # Print the image array
    #print(np.array(img))
    # get the max value in the image
    print(np.max(np.array(img)))
    # Print unique values in the image
    print(np.unique(np.array(img)))


    # img_path = os.path.join(root, 'resized',  'gt_binary_image', '0001.png')
    # img = cv2.imread(img_path)
    # img = Image.fromarray(img)
    # print(img.size)
    # print(img.mode)
    # # Print the image array
    # #print(np.array(img))
    # # get the max value in the image
    # print(np.max(np.array(img)))
    # # Print unique values in the image
    # print(np.unique(np.array(img)))

    img_path = os.path.join(root, 'resized',  'gt_instance_image', 'inst_0.png')
    img = Image.open(img_path)
    print(img.size)
    print(img.mode)
    print(img.getbands())
    print(img)
    # Print the image array
    #print(np.array(img))
    # get the max value in the image
    print(np.max(np.array(img)))
    # Print unique values in the image
    print(np.unique(np.array(img)))
    print(np.array(img))
    # Convert to binary image




    # img_path = os.path.join(root, 'resized',  'gt_instance_image', '0001.png')
    # img = cv2.imread(img_path)
    # img = Image.fromarray(img)
    # print(img.size)
    # print(img.mode)
    # # Print the image array
    # #print(np.array(img))
    # # get the max value in the image
    # print(np.max(np.array(img)))
    # # Print unique values in the image
    # print(np.unique(np.array(img)))
    # # Convert to binary image


    img_path = os.path.join(root, 'resized',  'gt_image', 'image_0.png')
    img = cv2.imread(img_path)
    img = Image.fromarray(img)
    print(img.size)
    print(img.mode)
    # Print the image array
    #print(np.array(img))
    # get the max value in the image
    print(np.unique(np.array(img)))
    print(np.max(np.array(img)))
    print(np.array(img))
    # Print unique values in the image
    #print(np.unique(np.array(img)))


    # img_path = os.path.join(root, 'resized',  'gt_image', '0001.png')
    # img = cv2.imread(img_path)
    # img = Image.fromarray(img)
    # print(img.size)
    # print(img.mode)
    # # Print the image array
    # #print(np.array(img))
    # # get the max value in the image
    # print(np.unique(np.array(img)))
    # print(np.max(np.array(img)))
    # # Print unique values in the image
    # #print(np.unique(np.array(img)))

