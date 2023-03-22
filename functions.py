import cv2 as cv2
import numpy as np

def get_bb_from_mask(mask_path):
    ''' Return the bounding box coordinates that fits the best the mask'''
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
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

def get_bb_iou(x1, y1, w1, h1, x2, y2, w2, h2):

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1+w1, x2+w2)
    y_bottom = min(y1+h1, y2+h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    a1 = w1*h1
    a2 = w2*h2
    
    return intersection_area / float(a1 + a2 - intersection_area)

'''
####A rajouter dans le main.py
from functions import get_bb_from_mask, get_bb_score

#in fucntion

score = []
name = 'whateverthenameis'
i=0

while True: 
    frame = cv.imread('sequences-train/'+ name +'/' + name + '-'+ str(i).zfill(3) + '.bmp')
    #x,y,w,h,predictions_resampled, predictions = pFilter.transition_state(frame)

    mask_path = 'sequences-train/'+ name +'/' +name + '-'+ str(i).zfill(3) + '.png'

    xm, ym, wm, hm = get_bb_from_mask(mask_path)
    #score.append(x,y,w,h,xm, ym, wm, hm)



import matplotlib.pyplot as plt
plt.plot(score, title= "distance between bb center through iterations")
plt.show()
'''
def meanshift_function(video_name, alpha):
    cap = cv2.VideoCapture('output/'+video_name+'.mp4')
    mask_path = 'sequences-train/'+ video_name +'/'+ video_name + '-'+ str(1).zfill(3) + '.png'


    # take first frame of the video
    ret,frame = cap.read()

    # select initial location of window
    roi_coord = get_bb_from_mask(mask_path)
    #cv2.destroyAllWindows()

    # setup initial location of window
    # r,h,c,w - region of image
    #           simply hardcoded the values
    r,h,c,w = int(roi_coord[1]),int(roi_coord[3]),int(roi_coord[0]),int(roi_coord[2])
    track_window = (c,r,w,h)

    # # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    # Writer to save video file:
    # get cap properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width` to int
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height` to int
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frameSize = (width,height)

    score = []
    i=1
    # histogram updating parameter
    roi_hist_iter = roi_hist
    while(1):
        ret ,frame = cap.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist_iter,[0,180],1)

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # update reference histogram
            x,y,w,h = track_window
            mask_path = 'sequences-train/'+ video_name +'/'+ video_name + '-'+ str(i).zfill(3) + '.png'

            xm, ym, wm, hm = get_bb_from_mask(mask_path)
            score.append(get_bb_score(x,y,w,h, xm, ym, wm, hm))

            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist_iter = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist_iter,roi_hist_iter,0,255,cv2.NORM_MINMAX)
            # Updated histogram is proportional to the original and the recalculated in this frame by a factor of alpha
            roi_hist_iter = alpha*roi_hist + (1-alpha)*roi_hist_iter

            # draw it on image
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
            cv2.imshow('img2',img2)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break

        i+=1

    cv2.destroyAllWindows()
    cap.release()
    print('One meanshift have been performed')


    return score
