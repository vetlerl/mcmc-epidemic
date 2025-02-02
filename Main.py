# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:17:19 2025

@author: arman et paul
"""

from scipy.sparse import diags
import numpy as np
import numpy.linalg as npl

#Global variables
a=1
b=2

#D
def BuildD(T):
    k = [np.ones(T),-2*np.ones(T-1),np.ones(T-2)]
    offset = [0,-1,-2]
    D = diags(k,offset).toarray()/4
    return D
#U,Delta,V
def BuildUVDelta(D):
    U, Delta, Vt = np.linalg.svd(D, full_matrices=False)
    return U,Delta,Vt

#A
def BuildA(Delta,Vt):
    A = np.diag(Delta) @ Vt
    return A
#sh
def Buildsh(T,a,b):
    sh = np.zeros(T)
    a = 1
    b = 2
    sh[0] = (a-2*b)/4
    sh[1] = b/2
    return sh

#Simulation of a n-sample of Y
def Computation_Y(T, Lambda):

    D=BuildD(T)
    
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    
    sh = Buildsh(T, a, b)
    
    
    x_tilde_true = np.zeros(T)
    for i in range(T) : # probably a more efficient way to do that
        rd = np.random.uniform(0,1)
        if(rd<0.3):
            x_tilde_true[i] = np.random.exponential(Lambda)

    x_true = npl.solve(D, x_tilde_true)
    Y = np.random.multivariate_normal(A@x_true, np.identity(T))
    
    return Y

#Compute argmax of pi and pi_tilde distributions
def ComputeArgmax(T,Lambda):
    D=BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    Y=Computation_Y(T, Lambda)
    
    u=U@Y+sh
    x_tilde=np.sign(u)*np.maximum(np.abs(u)-Lambda,np.zeros(T))-sh
    x=npl.solve(D,x_tilde)
    return x,x_tilde

#Test of the different functions 
D=BuildD(20)
U, Delta, Vt = BuildUVDelta(D)
A = BuildA(Delta, Vt)

sh = Buildsh(4, a, b)
x1,x2=ComputeArgmax(20,1)
