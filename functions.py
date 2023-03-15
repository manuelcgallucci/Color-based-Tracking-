import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

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

def meanshift_function(video_name, show_video = True):
    cap = cv2.VideoCapture('output/'+video_name+'.mp4')
    mask_path = 'sequences-train/'+ video_name +'/'+ video_name + '-'+ str(1).zfill(3) + '.png'


    # take first frame of the video
    ret,frame = cap.read()

    # select initial location of window
    roi_coord = get_bb_from_mask(mask_path)
    r,h,c,w = int(roi_coord[1]),int(roi_coord[3]),int(roi_coord[0]),int(roi_coord[2])
    track_window = (c,r,w,h)
    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


    ## INITIALISATION
    roi = frame[r:r+h, c:c+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 25., 50.)), np.array((180.,255.,255.)))
    init_roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(init_roi_hist,init_roi_hist,0,255,cv2.NORM_MINMAX)

    score = []
    hist_list = []
    roi_hist_iter = init_roi_hist
    hist_list.append(roi_hist_iter)
    i = 1


    while(1):
        ret ,frame = cap.read()

        if ret == True:
            # MEANSHIFT
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],init_roi_hist,[0,180],1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            x,y,w,h = track_window
            # Try to instead of doing a ponderate mean of the 2 histograms, do a translation of the median / search for MAP (variance pondered by the median)

            
            ## UPDATING THE HISTOGRAM
            
            
            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 12.,25.)), np.array((180.,255.,255.)))
            roi_hist_iter = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist_iter,roi_hist_iter,0,255,cv2.NORM_MINMAX)
            # Updated histogram is proportional to the original and the recalculated in this frame by a factor of alpha
            #roi_hist_iter = alpha*init_roi_hist + (1-alpha)*roi_hist_iter
            hist_list.append(roi_hist_iter)
            
            ## DRAWING
            if show_video:
                img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                cv2.imshow('Image video',img2)
            
            ## SCORE
            mask_path = 'sequences-train/'+ video_name +'/'+ video_name + '-'+ str(i).zfill(3) + '.png'
            xm, ym, wm, hm = get_bb_from_mask(mask_path)
            score.append(get_bb_score(x,y,w,h, xm, ym, wm, hm))


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break

        i+=1

    cv2.destroyAllWindows()
    cap.release()
    print('Meanshift have been performed for ' + video_name)

    return score , hist_list

'''
names = ['bag', 'bear', 'book', 'camel', 'rhino', 'swan']
with plt.style.context('bmh'):
    fig, ax = plt.subplots(2,3, figsize=(20,10))
    fig.suptitle('Histogram for each last bounding box', fontsize = 20)
    for i,name in enumerate(names):
        score, hist_list = meanshift_function(name, False)
        ax[i%2, i%3].set_title(name)
        #ax[i%2, i%3].axis('off')
        ax[i%2, i%3].plot(hist_list[-1])

    fig.savefig('./Plots/histogram_visual_last.jpg')
plt.show()
'''
