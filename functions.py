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
def change_hist(old_hist, new_hist, alpha):
    n = len(old_hist)
    old_peak = np.argmax(old_hist)
    new_peak = np.argmax(new_hist)
    peak = int(old_peak * alpha + new_peak * (1-alpha))
    hist = np.zeros(n)

    for val in old_hist:
        if old_peak > peak:
            for i in range(n-peak):         #on décale l'histogramme, des valeurs sont supprimées et les autres resteront à 0
                hist[i] = old_hist[i+peak]
        if old_peak < peak:
            for i in range(peak, n):
                hist[i] = old_hist[i-peak]
        if old_peak == peak:
            hist = np.copy(old_hist)
    
    return hist


def meanshift_function(video_name, alpha=0.9, show_video = True, ):
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
    hist_peak = []
    hist_iter = init_roi_hist
    hist_peak.append(np.argmax(hist_iter))
    i = 1


    while(1):
        ret ,frame = cap.read()

        if ret == True:
            # MEANSHIFT
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],hist_iter,[0,180],1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            x,y,w,h = track_window
            # Try to instead of doing a ponderate mean of the 2 histograms, do a translation of the median / search for MAP (variance pondered by the median)

            
            ## UPDATING THE HISTOGRAM
            
            
            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 12.,25.)), np.array((180.,255.,255.)))
            new_hist_iter = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(new_hist_iter,new_hist_iter,0,255,cv2.NORM_MINMAX)

            # Updated histogram is proportional to the original and the recalculated in this frame by a factor of alpha
            hist_iter = change_hist(hist_iter, new_hist_iter, alpha)
            

            hist_peak.append(np.argmax(hist_iter))
            
            ## DRAWING
            if show_video:
                img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                cv2.imshow('Image video',img2)
            

            ## SCORE
            print(i)
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

    return score , hist_peak


names = ['bag', 'bear', 'book', 'camel', 'rhino', 'swan']
with plt.style.context('bmh'):
    fig, ax = plt.subplots(2,3, figsize=(20,10))
    fig.suptitle('Hue of the peak value and score through images', fontsize = 20)
    for i,name in enumerate(names):
        score, hist_peak = meanshift_function(name, 0.9,False)
        ax[i%2, i%3].set_title(name)
        #ax[i%2, i%3].axis('off')
        ax[i%2, i%3].plot(hist_peak, color='b', label='Peak hue value')
        ax[i%2, i%3].legend()
        ax2 = ax[i%2, i%3].twinx()
        ax2.plot(score, color='r', label='Score')
        ax2.legend()



    fig.savefig('./Plots/histogram_score&peakhue_with_changing.jpg')
plt.show()

            

    
