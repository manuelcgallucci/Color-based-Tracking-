import cv2
import numpy as np
import matplotlib.pyplot as plt
from color_based_tracking import ParticleFilter, calculate_hsv_hist

# Todo use HSV channels (take the mean)
# Correct the boundries of the color based so we dont go out 

def plot_color_hist(histograms):
    # define colors to plot the histograms
    colors = ('k','g','c')
    names = ("h", 's', 'v')
    # compute and plot the image histograms
    for i,color in enumerate(colors):
        plt.plot(histograms[:,i],color=color, label=names[i])
    plt.legend()
    plt.title('Image Histogram ' + names[0]+names[1]+names[2])
    plt.show()

def main(hist_size=64):
    # ==== Initialization ===
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    window_size = (int(cap.get(3)), int(cap.get(4)))
    print(window_size)
    if not ret:
        print("Camera not found")
        return
    
    roi_coord = cv2.selectROI(frame)
    cv2.destroyAllWindows()

    roi_cropped = frame[int(roi_coord[1]):int(roi_coord[1]+roi_coord[3]), int(roi_coord[0]):int(roi_coord[0]+roi_coord[2])]

    # roi_hist = calculate_hsv_hist(roi_cropped, hist_size)
    # plot_color_hist(roi_hist)
    print("Coords:", roi_coord)
    
    # Define the intial conditions of the model
    pFilter = ParticleFilter(roi_coord[1], roi_coord[0], roi_coord[3], roi_coord[2], roi_cropped, \
            window_size=window_size, n_particles=100, dt=0.01, sigma=[10,10,1,1], hist_size=64, lambda_=20)
    
    # ==== Loop ===
    while(True):        
        ret, frame = cap.read()
        
        x,y,w,h,predictions_resampled, predictions = pFilter.transition_state(frame)

        cv2.rectangle(frame,(roi_coord[1], roi_coord[0]),(roi_coord[1]+roi_coord[3], roi_coord[0]+roi_coord[2]),(255,255,0),thickness=2)
        cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),thickness=2)

        for n in range(pFilter.n_particles):
            cx = int(predictions[n,0] + predictions[n,2] / 2.0)
            cy = int(predictions[n,1] + predictions[n,3] / 2.0)
            cv2.circle(frame, (cx,cy), radius=4, color =(0,255,0), thickness=-1)

        for n in range(pFilter.n_particles):
            cx = int(predictions_resampled[n,0] + predictions_resampled[n,2] / 2.0)
            cy = int(predictions_resampled[n,1] + predictions_resampled[n,3] / 2.0)
            cv2.circle(frame, (cx,cy), radius=1, color =(255,0,0), thickness=-1)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          


    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()