import cv2

def meanshift(frame, track_window, roi_hist, term_crit):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    x,y,w,h = track_window

    return x,y,w,h