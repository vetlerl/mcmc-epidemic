# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:17:19 2025

@author: arman et paul
"""

import scipy.stats
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

def ComputeMeans(T, Lambda):
    D=BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    sh = Buildsh(T, a, b)
    Y=Computation_Y(T, Lambda)
    mu_plus = U @ Y + Lambda*np.ones(T)
    C_plus = scipy.stats.norm.ppf(-sh-mu_plus)
    mu_minus = U @ Y - Lambda*np.ones(T)
    C_minus = 1 - scipy.stats.norm.ppf(-sh-mu_minus)
    gamma = C_plus / (C_plus + C_minus)
    mu_tilde_plus = mu_plus - np.exp(-((sh+mu_plus)**2)/2) / (np.sqrt(2*np.pi)*C_plus)
    mu_tilde_minus = mu_minus + np.exp(-((sh+mu_minus)**2)/2) / (np.sqrt(2*np.pi)*C_minus)
    mu_tilde = gamma*mu_tilde_plus + (1-gamma)*mu_tilde_minus
    mu = npl.solve(D, mu_tilde)
    return mu, mu_tilde

def ComputeQuantiles(T, Lambda, s):
    D=BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    sh = Buildsh(T, a, b)
    Y=Computation_Y(T, Lambda)
    mu_plus = U @ Y + Lambda*np.ones(T)
    C_plus = scipy.stats.norm.ppf(-sh-mu_plus)
    mu_minus = U @ Y - Lambda*np.ones(T)
    C_minus = 1 - scipy.stats.norm.ppf(-sh-mu_minus)
    gamma = C_plus / (C_plus + C_minus)
    probas = np.zeros(T)
    q_plus = np.zeros(T)
    q_minus = np.zeros(T)
    for i in range(T):
        q_plus[i] = scipy.stats.norm.ppf(min(s[i]-mu_plus[i], -sh[i]-mu_plus[i])) / C_plus[i]
        if(s[i]>-sh[i]):
            q_minus[i] = (scipy.stats.norm.ppf(s[i]-mu_minus[i]) + C_minus[i] - 1) / C_minus[i]
        probas[i] = gamma[i]*q_plus[i] + (1 - gamma[i])*q_minus[i]
    return probas

def DistributionPi(x, Y, A, D, sh, Lambda):
    return np.exp(((-1/2)*np.linalg.norm(Y - A@x)**2)-Lambda*sum(abs(D@x + sh)))

def MetropolisHastingsMean(T, Lambda):
    D=BuildD(T)
    gamma_zero = 0.01
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    Y=Computation_Y(T, Lambda)
    theta = np.ones(T) # maybe choose another starting point
    sum_theta = theta
    for i in range(10000):
        new_theta = np.random.multivariate_normal(theta, gamma_zero*np.identity(T))
        alpha = DistributionPi(new_theta, Y, A, D, sh, Lambda)*scipy.stats.norm.pdf()
    
#Test of the different functions 
D=BuildD(20)
U, Delta, Vt = BuildUVDelta(D)
A = BuildA(Delta, Vt)

sh = Buildsh(4, a, b)
x1,x2=ComputeArgmax(20,1)
