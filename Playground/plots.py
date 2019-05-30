from skimage.transform import radon, rescale, iradon
from skimage.io import imread

import matplotlib.use
import matplotlib.pyplot as plt

matplotlib.use('svg')


image = imread('data/phantom.png', as_gray=True)
plt.imshow(image)
plt.savefig()