'Define log pdfs here'
import numpy as np

def lognorm_pdf(xi_hat, xi, beta, Cov):
    x = (xi_hat - np.sqrt(1-beta**2) * xi)
    if xi.ndim==1:
        return - np.sum(xi_hat) - np.inner(x, Cov.inv(x)) / 2
    return - np.sum(xi_hat) - np.tensordot(x.T, Cov.inv(x)) / 2
    

def exp_pdf(x, beta):
    return -x/beta - np.log(beta)

def log_acceptance_ratio(xi, xi_hat, beta, Cov):
    return (
        lognorm_pdf(xi_hat, xi, beta, Cov)
         - lognorm_pdf(xi, xi_hat, beta, Cov) 
         + lognorm_pdf(xi_hat, np.zeros_like(xi_hat), np.ones_like(xi_hat), Cov) 
         - lognorm_pdf(xi, np.zeros_like(xi), np.ones_like(xi), Cov)
    )

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from operators import CovOp
    x = np.linspace(0.0001,10,100)
    C = CovOp(0,1)
    for i, e in enumerate(x):
        x[i] = lognorm_pdf(e, 1, .5, C)
    plt.plot(np.exp(x))
    plt.show()
    
    