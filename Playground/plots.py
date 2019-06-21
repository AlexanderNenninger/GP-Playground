from skimage.transform import radon, rescale, iradon
from skimage.io import imread

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pickle as pkl

# matplotlib.use('pdf')

fieldNames = ('samples', 'accepted', 'av_acc', 'xi', 'log_probs', 'betas', 'lengthscales' , 'proposals')
fName = 'output\\2019-06-04T12-15-51_n50000_accProb23%.pkl'

with open(fName, 'rb') as f:
    data = pkl.load(f)
    data = dict(zip(fieldNames, data))
pass


# image = imread('data/phantom.png', as_gray=True)
# plt.imshow(image)
# plt.savefig('plots/phantom')
# plt.clf()