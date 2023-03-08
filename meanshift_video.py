import numpy as np
import cv2
import glob

# name of the video sequence to study
video_name = str(input("Enter filename of video sequence to process: "))
# parameter to determine the rate of change of the reference histogram (lower = less change)
alpha = 0.1

cap = cv2.VideoCapture('output/'+video_name+'.mp4')

# take first frame of the video
ret,frame = cap.read()

# select initial location of window
roi_coord = cv2.selectROI(frame)
cv2.destroyAllWindows()

# setup initial location of window
# r,h,c,w - region of image
#           simply hardcoded the values
r,h,c,w = int(roi_coord[1]),int(roi_coord[3]),int(roi_coord[0]),int(roi_coord[2])
track_window = (c,r,w,h)

# # set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
print("fps:",fps)
print("frameSize",frameSize)

writer = cv2.VideoWriter('./output/'+video_name+'_meanshift.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)

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

        # write frame to output
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()
writer.release()

