import tensorflow as tf
import numpy as np
from mayavi import mlab

sess = tf.InteractiveSession()

theta = tf.placeholder(np.float32, shape=())
mu = tf.placeholder(np.float32, shape=(3,))
sigma = tf.placeholder(np.float32, shape=())

dWt = tf.random_normal([3])

eps = tf.placeholder(np.float32, shape=())

x_0 = np.asarray([1.,1.,0.], dtype=np.float32)

x = tf.Variable(x_0)

dx = theta * (mu - x) * eps + sigma * dWt * eps

step = tf.group(x.assign_add(dx))

tf.global_variables_initializer().run()


for i in range(1,10000):
    step.run({eps: .001, theta: 1, mu: [0.,0., 0.], sigma: 10 })
    _x = x.eval()
    try:
        xx = np.vstack((xx, _x[0]))
        yy = np.vstack((yy, _x[1]))
        zz = np.vstack((zz, _x[2]))
    except NameError:
        xx = _x[0]
        yy = _x[1]
        zz = _x[2]
mlab.plot3d(xx,yy,zz, tube_radius= .01, line_width = .1, color=(1,0,0))
mlab.show()