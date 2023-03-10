import numpy as np
import cv2

def calculate_hsv_hist(bgr_img, hist_size):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # Limit H bigger than 0.1 and S bigger than 0.2, No limit for V
    # mask = cv2.inRange(hsv_img, (25, 50, 0), (255, 255, 255))

    roi_hist = np.zeros((hist_size,3))
    for i in range(3):
        roi_hist[:,i] = cv2.calcHist(hsv_img, [i+1], None, [hist_size], [0,hist_size])[:,0]
        roi_hist[:,i] = roi_hist[:,i] / np.sum(roi_hist[:,i])
    # cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    return roi_hist

class ParticleFilter(object):
    def __init__(self, x0, y0, w0, h0, first_roi_bgr, window_size, n_particles=10, dt=1, sigma=[1,1,0.1,0.1], hist_size=64, lambda_=10, \
                min_size=40, max_size=200):
        # Define the initial state of the system
        self.state = np.array([x0, y0, w0, h0])
        self.n_state = self.state.shape[0]
        # Define the model matrices 
        self.A = np.array([[1+dt, 0, 0, 0], [0, 1+dt, 0, 0], [0, 0, 1+dt, 0], [0, 0, 0, 1+dt]])
        self.B = np.array([[-dt, 0, 0, 0], [0, -dt, 0, 0], [0, 0, -dt, 0], [0, 0, 0, -dt]])
        # self.C = np.zeros(0)

        # Define the noise sigma
        self.sigma_v = np.array(sigma)

        # Define all particles in the initial state
        self.particles = np.array([self.state,]*n_particles)
        self.last_particles = np.zeros(self.particles.shape)
        self.n_particles = n_particles
        
        # Define the first histogram
        self.hist = calculate_hsv_hist(first_roi_bgr, hist_size)
        self.hist_size = hist_size
        self.lambda_ = lambda_

        self.min_size = min_size
        self.max_size = max_size
        self.window_size = window_size

    def transition_state(self, frame):
        state_pred = self.state_prediction()
        
        state_pred = self.correct_size(state_pred)

        histograms = self.calculate_hist(state_pred, frame)
        weights = self.compare_hist(histograms, self.hist)

        self.last_particles = self.particles
        self.particles = self.resample(state_pred, weights)

        self.state = np.mean(self.particles, axis=0)
        roi_bgr = frame[int(self.state[0]):int(self.state[0]+self.state[2]), int(self.state[1]):int(self.state[1]+self.state[3])]
        self.hist = calculate_hsv_hist(roi_bgr, self.hist_size)


        return self.state[0],self.state[1],self.state[2], self.state[3], self.particles, state_pred
    
    def resample(self, predictions, weights):
        pos_indeces = np.arange(self.n_particles)
        indeces = np.random.choice(pos_indeces, self.n_particles, p=weights)
        return predictions[indeces]
    
    def compare_hist(self, histograms, histogram_ref):
        
        weights = np.zeros((self.n_particles))
        # For the moment only use the first dimension of the histogram
        for m in range(self.n_particles):
            weights[m] = np.exp(self.lambda_ * np.sqrt(np.sum(histograms[m,:,0]* histogram_ref[:,0])))
        
        return weights / np.sum(weights)

    def calculate_hist(self, state_pred, frame):
        histograms = np.zeros((self.n_particles, self.hist_size, 3))
        for k, state in enumerate(state_pred):
            # Crop roi into the frame
            # state_pred is: (x,y, w, h)
            init_x = min([self.window_size[0], max([0, int(state[0])])])
            max_x = min([self.window_size[0], max([1, int(state[0] + state[2])])])
            init_y = min([self.window_size[1], max([0, int(state[1])])])
            max_y = min([self.window_size[1], max([1, int(state[1] + state[3])])])
            
            
            # print(init_x, max_x, init_y, max_y)
            bgt_roi_cropped = frame[init_x:max_x, init_y:max_y]
            # print('\t', bgt_roi_cropped.shape, state)
            # Calculate and append the histogram
            histograms[k,:,:] = calculate_hsv_hist(bgt_roi_cropped, self.hist_size)
        return histograms

    def state_prediction(self):
        noise = self.sigma_v*np.random.randn(self.n_particles,self.n_state)
        particles = np.dot(self.particles, self.A) + np.dot(self.last_particles, self.B) + noise
        return particles
    
    def correct_size(self, predictions):
        """ Correct the size so the bb remains in the image"""
        # np.clip(predictions[:,0],self.state[2]+1,self.window_size[0]-(1+self.state[2]),predictions[:,0])        
        # np.clip(predictions[:,1],self.state[2]+1,self.window_size[1]-(1+self.state[2]),predictions[:,1])
        np.clip(predictions[:,2], self.min_size, self.max_size,predictions[:,2])
        np.clip(predictions[:,3], self.min_size, self.max_size,predictions[:,3])
        return predictions