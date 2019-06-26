import matplotlib.pyplot as plt
import numpy as np
import GPnd 

def plot_result_2d(image, chain, means, C, size, data, fbb):
    fig, ax = plt.subplots(2,4)

    ax[0,0].imshow(image)
    ax[0,0].scatter(size * means[:,0], size * means[:,1], c="r")
    ax[0,0].set_title('Original Image')

    ax[0,1].imshow(C.C[C.shape[0]//2,C.shape[1]//2])
    ax[0,1].set_title('Slice through the Covariance Operator')
    
    ax[0,2].plot(data)

    ax[0,3].imshow(fbb)


    ax[1,0].imshow(chain.reconstruction)
    ax[1,0].set_title('Reconstruction')
    ax[1,1].imshow(np.sqrt(chain.var))
    ax[1,2].imshow(chain.heightscale)
    ax[1,3].plot(chain.betas)
    plt.show()

if __name__=='__main__':
    pass
    # fig, ax = plt.subplots(2)
    # for t in T:
    #     ax[0].plot(t.F)
    # ax[1].plot(np.linspace(0,1, size), u)
    # ax[1].plot(means, y, 'r*')
    # plt.show()

    
    # print('acc prob ', np.mean(probs, axis=0))
    # print('2nd Layer:', np.mean([x[1] for x in samples]), end='	')

    # mean = np.mean([x[0] for x in samples], axis=0)
    # std = np.std([x[0] for x in samples], axis=0)

    # fig, ax = plt.subplots(2, 2)

    # ax[0,0].plot(mean)
    # ax[0,0].plot(mean + std, 'g--')
    # ax[0,0].plot(mean - std, 'g--')

    # ax[0,1].plot(betas)

    # ax[1,0].plot([x[1] for x in samples])
    # ax[1,1].hist([x[1] for x in samples], density=True, bins = 20)