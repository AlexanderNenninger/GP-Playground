from skimage.transform import radon, rescale, iradon
from skimage.io import imread

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pdf')

image = imread('data/phantom.png', as_gray=True)
plt.imshow(image)
plt.savefig('plots/phantom')
plt.clf()