import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from color_based_tracking import ParticleFilter, calculate_hsv_hist
from functions import get_bb_from_mask, get_bb_score
from meanshift import meanshift

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

def main(dir_imgs, dir_masks, video_name, type_='probabilistic', hist_size=180, show=False):
	'''
	dir_imgs and dir_masks with '/' at the end
	type_ = 'probabilistic' or 'meanshift'
	'''
	# ==== Initialization ===
	filename = glob.glob(dir_imgs + video_name +'/*.bmp')[0]
	frame = cv2.imread(filename)
	height, width, layers = frame.shape
	frameSize = (width,height)

	# select initial location of ROI
	mask_path = dir_masks + video_name +"/"+ video_name + '-'+ str(1).zfill(3) + '.png'
	roi_coord = get_bb_from_mask(mask_path)
	# roi_coord: is x,y,w,h
	x,y,w,h = int(roi_coord[0]),int(roi_coord[1]),int(roi_coord[2]),int(roi_coord[3])
	track_window = (x,y,w,h)
	roi = frame[y:y+h, x:x+w]

	# Initialize performance metrics and outputs
	score = []
	centroids = []
	centroids = np.array(centroids)
	fps = 24
	video_writer = cv2.VideoWriter('./output/{:s}/{:s}_{:s}.mp4'.format(type_, video_name, type_), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)

	# Define the intial conditions of the model
	if type_== 'probabilistic':

		pFilter = ParticleFilter(roi_coord[0], roi_coord[1], roi_coord[2], roi_coord[3], roi, \
            window_size=frameSize, n_particles=1000, dt=0.2, sigma=[5,5,0.5,0.5], hist_size=16, lambda_=20, 
            min_size_x=int(roi_coord[2]*0.8), max_size_x=int(roi_coord[2]*1.2),
            min_size_y=int(roi_coord[3]*0.8), max_size_y=int(roi_coord[3]*1.2))
	
	elif type_=='meanshift':
		# Setup the ROI for tracking
		hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_roi, np.array((0., 25., 50.)), np.array((180.,255.,255.)))
		init_roi_hist = cv2.calcHist([hsv_roi],[0],mask,[hist_size],[0,hist_size])
		cv2.normalize(init_roi_hist,init_roi_hist,0,255,cv2.NORM_MINMAX)

		# Setup the termination criteria, either 10 iteration or move by at least 1 pt
		term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

		roi_hist_iter = init_roi_hist

	# ==== Loop ===
	for i, filename in enumerate(glob.glob(dir_imgs + video_name +'/*.bmp')):
		# Skip first image
		# if i == 0:
		# 	continue
		
		# Read frame
		frame = cv2.imread(filename)
		
		# Predict bounding box
		if type_== 'probabilistic':
			x,y,w,h,predictions_resampled, predictions, weights = pFilter.transition_state(frame)
		elif type_== 'meanshift':
			track_window, roi_hist_iter = meanshift(frame, track_window, init_roi_hist, roi_hist_iter, term_crit, alpha=1)
			x,y,w,h = track_window
			# print(track_window)
		
		# Compare to mask and append score
		mask_path = dir_masks + video_name +"/"+ video_name + '-'+ str(i+1).zfill(3) + '.png'
		xm, ym, wm, hm = get_bb_from_mask(mask_path)
		score.append(get_bb_score(x,y,w,h, xm, ym, wm, hm))
		# TODO: add output centroids.npy
		centroid = (xm+wm/2,ym+hm/2)
		np.append(centroids, centroid)
		
		# Draw rectangle
		cv2.rectangle(frame,(int(xm),int(ym)),(int(xm+wm),int(ym+hm)),(0,255,0),thickness=2)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), thickness=2)

		# write frame to output
		video_writer.write(frame)
		
		if show: cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
	#  plot_color_hist(roi_hist_iter)
	video_writer.release()
	
	#  save centroids.npy output
	np.save("./output/centroids_{:s}_{:s}".format(video_name, type_), centroids)
	
	#  plot score
	plt.figure()
	plt.plot(score)
	plt.title("Score squence:{:s} {:s}".format(video_name, type_))
	plt.xlabel("Frames")
	if show: plt.show()
	plt.savefig("./output/{:s}/{:s}_score_{:s}.png".format(type_,video_name, type_))
	plt.close()

if __name__=="__main__":
	dir_imgs = "./sequences-train/"
	dir_mask = "./sequences-train/"
	video_names = ["bag", "book", "bear", "bag", "camel", "rhino", "swan"]
	type_ = "meanshift"
	show = False
	hist_size = 180
	
	for video_name in video_names:		
		main(dir_imgs, dir_mask, video_name, type_=type_, show=show)
