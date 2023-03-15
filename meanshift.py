def meanshift(frame, roi_hist, roi_hist_iter, term_crit, alpha=0.9)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist_iter,[0,180],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    x,y,w,h = track_window

    # Calculate 
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist_iter = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist_iter,roi_hist_iter,0,255,cv2.NORM_MINMAX)

    # Updated histogram is proportional to the original and the recalculated in this frame by a factor of alpha
    roi_hist_iter = alpha*roi_hist + (1-alpha)*roi_hist_iter

    return x,y,w,h , roi_hist_iter