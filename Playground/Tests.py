import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,50)

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1

    xx, yy = np.mgrid[xmin:xmax:9j, ymin:ymax:9j]
    X = np.vstack([xx.ravel(), yy.ravel()]).T

    f = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])