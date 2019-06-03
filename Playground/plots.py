from skimage.transform import radon, rescale, iradon
from skimage.io import imread

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pickle as pkl

fName = 'output\\2019-22-7_niter50000_accProb24%.pkl'
with open(fName, 'rb') as f:
    data = pkl.load(f)

# matplotlib.use('pdf')

# image = imread('data/phantom.png', as_gray=True)
# plt.imshow(image)
# plt.savefig('plots/phantom')
# plt.clf()