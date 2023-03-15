import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from color_based_tracking import ParticleFilter, calculate_hsv_hist
from functions import get_bb_from_mask, get_bb_score

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

def main(dir_imgs, dir_masks, video_name, type_='probabilistic', hist_size=64):
	'''
	dir_imgs and dir_masks with '/' at the end
	type_ = 'probabilistic' or 'meanshift'
	'''
    # ==== Initialization ===
    print(glob.glob(dir_imgs + video_name +'/*.bmp'))
    filename = glob.glob(dir_imgs + video_name +'/*.bmp')[0]
    img = cv2.imread(filename)
	height, width, layers = img.shape
	frameSize = (width,height)
	
    # select initial location of ROI
	roi_coord = get_bb_from_mask(mask_path)
	r,h,c,w = int(roi_coord[1]),int(roi_coord[3]),int(roi_coord[0]),int(roi_coord[2])
	track_window = (c,r,w,h)
	roi = frame[r:r+h, c:c+w]
	
	# Initialize performance metrics and outputs
	score = []
	fps = 24
	writer = cv2.VideoWriter('./output/'+video_name+'_meanshift.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)
	
    # roi_hist = calculate_hsv_hist(roi, hist_size)
    # plot_color_hist(roi_hist)
    print("Coords:", roi_coord)
    
	# Define the intial conditions of the model
	if type_== 'probabilistic':
		pFilter = ParticleFilter(roi_coord[1], roi_coord[0], roi_coord[3], roi_coord[2], roi, \
	        window_size=frameSize, n_particles=100, dt=0.01, sigma=[10,10,1,1], hist_size=hist_size, lambda_=20)
	
	# ==== Loop ===
	for i, filename in enumerate(glob.glob(dir_imgs + video_name +'/*.bmp')):
	    # Read frame
	    frame = cv2.imread(filename)
	    
	    # Predict bounding box
	    if type_== 'probabilistic':
	    	x,y,w,h,predictions_resampled, predictions = pFilter.transition_state(frame)
	    else if type_== 'meanshift':
	    	pass
		
		# Compare to mask and append score
		mask_path = dir_masks + video_name + '-'+ str(i).zfill(3) + '.png'
		xm, ym, wm, hm = get_bb_from_mask(mask_path)
    	score.append(get_bb_score(x,y,w,h, xm, ym, wm, hm))
    	# TODO: add output centroids.npy
    	
    	# Draw rectangle
	    cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),thickness=2)
	    
	    # write frame to output
	    writer.write(frame)
	    
	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
            break      
	
	writer.release()
	

if __name__=="__main__":
    main()
