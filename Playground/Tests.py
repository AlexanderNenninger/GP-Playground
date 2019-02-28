from mayavi import mlab
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import Kernel

ker = Matern()

X = np.matrix([0.]*100)
for i in range(1,10):
        X = np.vstack((X,np.matrix([i]*100)))

X_eval = np.matrix([0.]*100)
for i in range(1,20):
    X_eval = np.vstack((X_eval,np.matrix([i]*100)))

C = ker(X, X_eval)
print(C@X_eval, C.shape)
mlab.surf(C)
mlab.show()