# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:17:19 2025

@author: arman
"""

from scipy.sparse import diags
import numpy as np

def Computation_Y(n, Lambda):

    k = [np.ones(n-1),-2*np.ones(n),np.ones(n-1)]
    offset = [-1,0,1]
    D = diags(k,offset).toarray()
    
    U, Delta, Vt = np.linalg.svd(D, full_matrices=False)
    
    A = np.diags(Delta) @ Vt
    
    sh = np.zeros(n)
    a = 1
    b = 2
    sh[0] = (a-2*b)/4
    sh[1] = b/2
    
    x_tilde_true = np.zeros(n)
    for i in range(n) : # probably a more efficient way to do that
        rd = np.random.uniform(0,1)
        if(rd<0.3):
            x_tilde_true[i] = np.random.exponential(Lambda)

    x_true = np.solve(D, x_tilde_true)
    Y = np.random.multivariate_normal(A@x_true, np.identity(n))
    
    return Y
    
    