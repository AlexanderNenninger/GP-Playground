import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale, iradon

x = np.linspace(0,1,50)

xmin, xmax = 0, 1
ymin, ymax = 0, 1
imgSize = 31j

xx, yy = np.mgrid[xmin:xmax:imgSize, ymin:ymax:imgSize]
X = np.vstack([xx.ravel(), yy.ravel()]).T

f = lambda xx, yy: np.sin(np.pi * xx) * np.sin(2 * np.pi * yy)
fX = f(xx, yy)

size = []
#asserted that len(sinogram) ~ len(image) -> dx ~ 1/size(image)
for i in range(10,100):
    img = np.random.standard_normal((i,i))
    sinogram = radon(img, [0, 20], circle=False)
    size.append(sinogram.shape[0])

print(size)
plt.plot(size)
plt.show()        