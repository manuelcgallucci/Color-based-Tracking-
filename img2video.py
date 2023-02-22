import cv2
import numpy as np
import glob

frameSize = (480,360)

out = cv2.VideoWriter('bag.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

for filename in glob.glob('sequences-train/bag/*.bmp'):
    img = cv2.imread(filename)
    out.write(img)

out.release()