#### Here I will test our mean-shift algorithm and search for the optimal parameters (alpha, threshold)
# Running time ~ 30 sec
# Note: before running this you have to run img2videos.py in order to create the mp4 needed

from functions import meanshift_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

alphas = np.arange(start=0.5, step=0.01, stop=1.01)
names = ['bag', 'bear', 'book', 'camel', 'rhino', 'swan']
scores = [[[] for i in range(len(alphas))] for j in range(len(names))]
score_mean = np.zeros((len(names), len(alphas)))

fig, ax = plt.subplots(len(names), len(alphas), sharey='row', figsize = (200,100))
for i, name in enumerate(names):
    ax[i, 0].set_ylabel(name)
    for j,alpha in enumerate(alphas):
        score = meanshift_function(name, alpha)
        scores[i][j] = score
        score_mean[i][j] = np.mean(score)
        ax[i,j].plot(score)

for i, alpha in enumerate(alphas):
    ax[0,i].set_title(alpha)


score_df = pd.DataFrame(score_mean).T

score_df.columns = names
score_df.index = alphas
score_df['mean'] = score_df.mean(axis=1)
print(score_df) 

plt.show()

plt.figure()
plt.plot(score_df['mean'])
plt.title('Evolution of the mean error on the test dataset with alpha')
plt.xlabel('alpha value')
plt.ylabel('Distance between centroids')
plt.show()