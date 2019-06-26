from pathlib import Path

from skimage.io import imread, imshow
from skimage.transform import rescale


def import_image(image_path, size=20):
    image = imread(image_path, as_gray=True)
    scale = tuple(size/s for s in image.shape)
    return rescale(image, scale, multichannel=False)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    image = import_image()
    plt.imshow(image, cmap='Greys_r')
    plt.show()