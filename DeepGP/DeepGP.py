import numpy as np
import matplotlib.pyplot as plt

def xi(m: np.array, C:np. array):
    dim = m.shape[0]
    return C @ np.random.randn(dim) + m

def phi(x):
    return 1*np.linalg.norm(x)**2

def I(u, sqrtC):
    return phi(u) + .5 * np.sum((sqrtC @ u)**2)

def a(u, v, sqrtC):
    return np.min([1, np.exp(I(u, sqrtC) - I(v, sqrtC))])

if __name__=='__main__':
    sqrtC = np.array(
        [
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ]
    )
    C = sqrtC@sqrtC
    m = np.array([0,0,0])


    beta = np.array([
        [.9,.9,.9],
        [.2,.2,.2],
        [.5,.5,.5]
        ])
    u = m
    
    print(I(u, sqrtC))


    path = []
    a_path = []
    for k in range(0,1000):

        
        v = u + beta @ xi(m, C)
        if a(u,v, sqrtC) <= np.random.rand():
            u = v
            path.append(u)
    plt.plot(path)
    plt.show()

    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = phi([xx,yy])
    plt.contourf(x,y,z)
    
    plt.show()