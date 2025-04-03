import scipy.stats as sps
from scipy.sparse import diags
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt

def PD3S(T, A, niter=1e5, a, b):
    
    theta = np.concatenate(np.ones(T),np.zeros(T))
    s = A@theta
    M = np.zeros(2*T)
    lambda_max = np.linalg.norm(D@(D.T), ord=2)
    Lmax = np.max(lambda_max, (lambda_O/lambda_R)^2 * np.max(c**2))
    gamma = 1/phi_order
    delta = 0.999/(gamma*Lmax)

    for k in range(niter):
        tau = s + delta*A@theta + delta*A@M
        s = tau + delta*sh - np.max(np.abs(tau + delta*sh) - lambda_R, np.zeros(2*T))*np.sign(tau + delta*sh)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    
    u=U@Y+sh
    x_tilde=np.sign(u)*np.maximum(np.abs(u)-Lambda,np.zeros(T))-sh
    x=npl.solve(D,x_tilde)
    return x,x_tilde