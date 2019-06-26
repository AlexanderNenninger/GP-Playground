from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import GPnd 

def plot_result_2d(image, chain, means, C, size, data, fbb):
    plt.switch_backend('pdf')
    fig, ax = plt.subplots(2,4)

    ax[0,0].imshow(image)
    #ax[0,0].scatter(size * means[:,0], size * means[:,1], c="r")
    ax[0,0].set_title('Original Image')

    ax[0,1].imshow(C.C[C.shape[0]//2,C.shape[1]//2])
    ax[0,1].set_title('Slice through the Covariance Operator')
    
    ax[0,2].plot(data)
    ax[0,2].set_title('Results of the Radon Transform')

    ax[0,3].imshow(fbb)
    ax[0,3].set_title('Reconstruction via Filtered Backprojections')

    ax[1,0].imshow(chain.reconstruction)
    ax[1,0].set_title('Reconstruction via MCMC')
    
    ax[1,1].imshow(np.sqrt(chain.var))
    ax[1,1].set_title('Standard deviation of the Samples')

    ax[1,2].plot([s[1] for s in chain.samples])
    ax[1,2].set_title('Heightscale')
    
    ax[1,3].plot(chain.betas)
    ax[1,3].set_title('Jump Size')
    plt.savefig('fig_%s.pdf'%datetime.now().replace(microsecond=0).isoformat().replace(':','-'))