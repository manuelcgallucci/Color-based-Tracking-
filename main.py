import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_color_hist(histograms):
    # define colors to plot the histograms
    colors = ('b','g','r')
    
    # compute and plot the image histograms
    for i,color in enumerate(colors):
        plt.plot(histograms[i,:],color = color)
    plt.title('Image Histogram GFG')
    plt.show()  

def main(hist_size=64):

    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    
    if not ret:
        print("Camera not found")
        return
    
    roi_coord = cv2.selectROI(frame)
    cv2.destroyAllWindows()

    roi_cropped = frame[int(roi_coord[1]):int(roi_coord[1]+roi_coord[3]), int(roi_coord[0]):int(roi_coord[0]+roi_coord[2])]
    cv2.imshow("ROI",roi_cropped)
    ref_hist = np.zeros((3,hist_size))
    for i in range(3):
        ref_hist[i,:] = cv2.calcHist(roi_cropped, [i+1],None, [hist_size], [0,hist_size])[:,0]
    
    plot_color_hist(ref_hist)

    while(True):        
        ret, frame = cap.read()
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()