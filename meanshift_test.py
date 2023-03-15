#### Here I will test our mean-shift algorithm and search for the optimal parameters (alpha, threshold)
# Running time depends on number of parameters ~ 30 sec
# Note: before running this you have to run img2videos.py in order to create the mp4 needed



#### REsULTS
# We saw that alpha != 1 isn't effiicent
from functions import meanshift_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



names = ['bag', 'bear', 'book', 'camel', 'rhino', 'swan']
scores = [[] for j in range(len(names))]
score_mean = np.zeros((len(names)))

fig, ax = plt.subplots(len(names), 1, sharey='row', figsize = (200,100))
for i, name in enumerate(names):
    ax[i].set_ylabel(name)
    
    score, hist_list = meanshift_function(name)
    scores[i] = score
    score_mean[i] = np.mean(score)
    ax[i].plot(score)


score_df = pd.DataFrame(score_mean).T

score_df.columns = names
score_df.index = ['Centroid distance']
score_df['mean'] = score_df.mean(axis=1)
print(score_df) 

plt.show()

plt.figure()
plt.plot(score_df['mean'])
plt.title('Evolution of the mean error on the test dataset with alpha')
plt.xlabel('alpha value')
plt.ylabel('Distance between centroids')
plt.show()