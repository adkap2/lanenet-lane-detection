import cv2
import numpy as np
import glob


def get_img_array(img_path):

    img_array = []

    filenames = []
    for filename in glob.glob(img_path):
        filenames.append(filename)
    filenames.sort(key=getint)

    for filename in filenames:
        img = cv2.imread(filename)
        img_array.append(img)
    
    return img_array


def make_video(img_array, video_path):

    """
    :param img_array: list of images
    :param video_path: path to save video"""

    size = (img_array[0].shape[1], img_array[0].shape[0])

    out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc('M','J','P','G'), 10, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()


def getint(name):
    basename = name.partition('.')
    num = basename[0].split('_')[-1]
    return int(num)

if __name__=="__main__":

    img_path = '/Users/adam/Desktop/SJSU/CMPE249/Project/lanenet-lane-detection/data/test_images_3/output/*.png'
    video_path = '/Users/adam/Desktop/SJSU/CMPE249/Project/lanenet-lane-detection/data/test_images_3/output.avi'

    img_array = get_img_array(img_path)


    make_video(img_array, video_path)
