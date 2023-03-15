from functions import get_bb_from_mask, get_bb_score
from color_based_tracking import ParticleFilter
import cv2
import matplotlib.pyplot as plt
import os

def main(name = 'bag'):
    score = []
    i=0
    print('./sequences-train/'+ name +'/'+ name + '-'+ str(i).zfill(3) + '.bmp')
    if not os.path.exists('./sequences-train/'+ name +'/'+ name + '-'+ str(i).zfill(3) + '.bmp'):
        return -1
    frame = cv2.imread('./sequences-train/'+ name +'/'+ name + '-'+ str(i).zfill(3) + '.bmp')
    mask_path = './sequences-train/'+ name +'/' +name + '-'+ str(i).zfill(3) + '.png'
    window_size = frame.shape[0:2]
    xm, ym, wm, hm = get_bb_from_mask(mask_path)

    pFilter = ParticleFilter(xm, ym, wm, hm, \
                window_size=window_size, n_particles=200, dt=0.1, sigma=[10,10,0.5,0.5], hist_size=32, lambda_=20, 
                min_size_x=int(wm*0.8), max_size_x=int(wm*1.2),
                min_size_y=int(hm*0.8), max_size_y=int(hm*1.2))

    i = i + 1
    while True: 
        if not os.path.exists('./sequences-train/'+ name +'/'+ name + '-'+ str(i).zfill(3) + '.bmp'):
            break
        frame = cv2.imread('./sequences-train/'+ name +'/'+ name + '-'+ str(i).zfill(3) + '.bmp')
        mask_path = './sequences-train/'+ name +'/' +name + '-'+ str(i).zfill(3) + '.png'
        
        x,y,w,h,predictions_resampled, predictions, weights = pFilter.transition_state(frame)
        score.append(x,y,w,h,xm, ym, wm, hm)

    plt.figure()
    plt.plot(score, title= "distance between bb center through iterations")
    plt.show()
    plt.close()
    return 1

if __name__ == "__main__":
    print(main())