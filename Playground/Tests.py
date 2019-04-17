import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

x = np.arange(0,1, .01)
y = np.sin(20*np.pi*x)

sampling_period = x[-1]-x[0]
scale = 1

coef, freqs = pywt.cwt(y,np.arange(1,129),'gaus1')
plt.matshow(coef)
plt.show()
plt.plot(freqs)
plt.show()