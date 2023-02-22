import cv2
import numpy as np
import glob

# Set specific video parameters:
video_sequence_name = 'bag'
dir_ = './sequences-train/'

img_array = []
for filename in glob.glob(dir_+ video_sequence_name +'/*.bmp'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    frameSize = (width,height)
    img_array.append(img)

assert 'frameSize' in globals(), "Images not found or not loaded correctly"

out = cv2.VideoWriter('./output/bag.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
