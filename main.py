import cv2
import numpy as np
import matplotlib.pyplot as plt
from color_based_tracking import ParticleFilter, calculate_hsv_hist

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
    
    if not ret:
        print("Camera not found")
        return
    
    roi_coord = cv2.selectROI(frame)
    cv2.destroyAllWindows()

    roi_cropped = frame[int(roi_coord[1]):int(roi_coord[1]+roi_coord[3]), int(roi_coord[0]):int(roi_coord[0]+roi_coord[2])]

    roi_hist = calculate_hsv_hist(roi_cropped, hist_size)
    plot_color_hist(roi_hist)

    # Define the intial conditions of the model
    pFilter = ParticleFilter(roi_coord[1], roi_coord[0], roi_coord[3], roi_coord[2], roi_cropped, n_particles=100, dt=1, sigma=[1,1,0.1,0.1], hist_size=64, lambda_=10)
    # ==== Loop ===
    while(True):        
        ret, frame = cap.read()
        
        x,y,w,h,predictions_resampled, predictions = pf.next_state(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          


    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()