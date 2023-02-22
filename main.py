import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def calculate_hsv_hist(bgr_img, hist_size):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # Limit H bigger than 0.1 and S bigger than 0.2, No limit for V
    mask = cv2.inRange(hsv_img, (25, 50, 0), (255, 255, 255))

    roi_hist = np.zeros((hist_size,3))
    for i in range(3):
        roi_hist[:,i] = cv2.calcHist(hsv_img, [i+1], mask, [hist_size], [0,hist_size])[:,0]
    # cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    return roi_hist

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
    
    # ==== Loop ===
    while(True):        
        ret, frame = cap.read()
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          
        break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()