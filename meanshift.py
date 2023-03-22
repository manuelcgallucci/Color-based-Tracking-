import cv2
import numpy as np

def meanshift(frame, track_window, init_roi_hist, roi_hist_iter, term_crit, alpha=1):
    # MEANSHIFT
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #print(roi_hist_iter[:,0])
    dst = cv2.calcBackProject([hsv],[0,1],roi_hist_iter[:,:1],[0,180],1)
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    ## UPDATING THE HISTOGRAM
    x,y,w,h = track_window
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 25.,50.)), np.array((180.,255.,255.)))
    roi_hist_iter = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist_iter,roi_hist_iter,0,255,cv2.NORM_MINMAX)

    # Updated histogram is proportional to the original and the recalculated in this frame by a factor of alpha
    roi_hist_iter = alpha*init_roi_hist #+ (1-alpha)*roi_hist_iter

    return track_window, roi_hist_iter