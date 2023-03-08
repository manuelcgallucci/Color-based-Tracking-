import cv2 as cv
import numpy as np

def get_bb_from_mask(mask_path):
    ''' Return the bounding box coordinates that fits the best the mask'''
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    
    xid = np.where(np.sum(mask, axis=0) != 0)
    x   = np.min(xid)
    w   = np.max(xid) - x
    yid = np.where(np.sum(mask, axis=1) != 0)
    y   = np.min(yid)
    h   = np.max(yid) -y
    
    return x,y,w,h

def get_bb_score(x1, y1, w1, h1, x2, y2, w2, h2):
    ''' Return the distance between centers of masks'''
    return  np.sqrt(((x1 + w1//2) - (x2 + w2//2))**2 + ((y1 + h1//2) - (y2 + h2//2))**2)




####A rajouter dans le main.py
from functions import get_bb_from_mask, get_bb_score

#in fucntion

score = []
name = 'whateverthenameis'
i=0

while True: 
    frame = cv.imread('sequences-train/'+ name + '-'+ str(i).zfill(3) + '.bmp')
    x,y,w,h,predictions_resampled, predictions = pFilter.transition_state(frame)

    mask_path = 'sequences-train/'+ name + '-'+ str(i).zfill(3) + '.png'

    xm, ym, wm, hm = get_bb_from_mask(mask_path)
    score.append(x,y,w,h,xm, ym, wm, hm)



import matplotlib.pyplot as plt
plt.plot(score, title= "distance between bb center through iterations")
plt.show()

