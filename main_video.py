import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from color_based_tracking import ParticleFilter, calculate_hsv_hist
from functions import get_bb_from_mask, get_bb_score


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

def main(dir_imgs, dir_masks, video_name, type_='probabilistic', hist_size=64, show=False):
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
	r,h,c,w = int(roi_coord[1]),int(roi_coord[3]),int(roi_coord[0]),int(roi_coord[2])
	# track_window = (c,r,w,h)
	roi = frame[r:r+h, c:c+w]

	# Initialize performance metrics and outputs
	score = []
	fps = 24
	video_writer = cv2.VideoWriter('./output/{:s}_{:s}.mp4'.format(video_name, type_), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)

	# roi_hist = calculate_hsv_hist(roi, hist_size)
	# plot_color_hist(roi_hist)
	print("Coords:", roi_coord)

	# Define the intial conditions of the model
	if type_== 'probabilistic':

		pFilter = ParticleFilter(roi_coord[0], roi_coord[1], roi_coord[2], roi_coord[3], roi, \
            window_size=frameSize, n_particles=1000, dt=0.2, sigma=[5,5,0.5,0.5], hist_size=16, lambda_=20, 
            min_size_x=int(roi_coord[2]*0.8), max_size_x=int(roi_coord[2]*1.2),
            min_size_y=int(roi_coord[3]*0.8), max_size_y=int(roi_coord[3]*1.2))
		
	# ==== Loop ===
	for i, filename in enumerate(glob.glob(dir_imgs + video_name +'/*.bmp')):
		if i == 0:
			continue
		# Read frame
		frame = cv2.imread(filename)
		
		# Predict bounding box
		if type_== 'probabilistic':
			x,y,w,h,predictions_resampled, predictions, weights = pFilter.transition_state(frame)
		elif type_== 'meanshift':
			pass
		
		# Compare to mask and append score
		mask_path = dir_masks + video_name +"/"+ video_name + '-'+ str(i+1).zfill(3) + '.png'
		print(mask_path)
		xm, ym, wm, hm = get_bb_from_mask(mask_path)
		score.append(get_bb_score(x,y,w,h, xm, ym, wm, hm))
		# TODO: add output centroids.npy
		
		
		
		# Draw rectangle
		cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),thickness=2)
		cv2.rectangle(frame,(int(xm),int(ym)),(int(xm+wm),int(ym+hm)),(0,255,0),thickness=2)
		# write frame to output
		video_writer.write(frame)
		
		if show: cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break      
	
	video_writer.release()

	plt.figure()
	plt.plot(score)
	plt.title("Score squence:{:s}".format(video_name))
	plt.xlabel("Frames")
	if show: plt.show()
	plt.savefig("{:s}_score.png".format(video_name))
	plt.close()

if __name__=="__main__":
	dir_imgs = "./sequences-train/"
	dir_mask = "./sequences-train/"
	video_names = ["bag", "book", "bear", "bag", "camel", "rhino", "swan"]
	type_ = "probabilistic"
	show = False
	
	for video_name in video_names:		
		main(dir_imgs, dir_mask, video_name, type_=type_, show=show)
