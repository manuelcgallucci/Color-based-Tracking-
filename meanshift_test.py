#### Here I will test our mean-shift algorithm and search for the optimal parameters (alpha, threshold)
# Running time depends on number of parameters ~ 30 sec
# Note: before running this you have to run img2videos.py in order to create the mp4 needed





from functions import meanshift_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# For getting performance 

# names = ['bag', 'bear', 'book', 'camel', 'rhino', 'swan'] # train set

names = ['cow', 'fish', 'octopus'] # test set

scores = [[] for j in range(len(names))]
score_mean = np.zeros((len(names)))

fig, ax = plt.subplots(len(names), 1, sharey='row', figsize = (200,100))
for i, name in enumerate(names):
    ax[i].set_ylabel(name)
    
    score, hist_list = meanshift_function(name, show_video=True)
    scores[i] = score
    score_mean[i] = np.mean(score)
    ax[i].plot(score)


score_df = pd.DataFrame(score_mean).T

score_df.columns = names
score_df.index = ['Centroid distance']
score_df['mean'] = score_df.mean(axis=1)
print(score_df) 

for i,name in enumerate(names):
    path = '8-'+name + '-centroids.npy'
    np.save(path, scores[i])



# For plotting the evolution of the score with alpha

'''
names = ['bag', 'bear', 'book', 'camel', 'rhino', 'swan']
alphas = np.arange(0, 1.1, 0.1)
score_mean = np.zeros((len(names), len(alphas)))
fig, ax = plt.subplots(len(names), len(alphas), sharey='row', figsize = (200,100))
for i, name in enumerate(names):
    ax[i,0].set_ylabel(name)
    for j, alpha in enumerate(alphas):
        score, hist_list = meanshift_function(name, alpha, False)
        score_mean[i,j] = np.mean(score)
        ax[i,j].plot(score)


score_df = pd.DataFrame(score_mean).T

score_df.columns = names
score_df.index = alphas
score_df['mean'] = score_df.mean(axis=1)
print(score_df)


with plt.style.context('bmh'):
    fig, ax = plt.subplots(1,1)
    ax.plot(score_df['mean'])
    ax.set_title('Evolution of the mean error on the test dataset with alpha')
    ax.set_xlabel('alpha value')
    ax.set_ylabel('Distance between centroids')
    fig.savefig('./Plots/histogram_score&peakhue_with_changing_only_int.jpg')

plt.show()
'''

#### REsULTS
# We saw that alpha != 1 isn't effiicent when it's only a ponderate mean of the 2 distributions
# Update: even when we translate progessively the hist, the better aren't best

names = ['cow', 'fish', 'octopus']
plt.figure()
for name in names:
    path = '8-'+name + '-centroids.npy'
    score = np.load(path)
    plt.plot(score, label=name)

plt.xlabel('nb of frame')
plt.ylabel('distance of centroids')
plt.legend()
plt.show()
