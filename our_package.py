# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:17:19 2025

@author: arman paul et vetle
"""

import scipy.stats as sps
from scipy.sparse import diags
import numpy as np
import numpy.linalg as npl
import numpy.random as npr

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
    U, Delta, Vt = npl.svd(D, full_matrices=False)
    return U,Delta,Vt

#A
def BuildA(Delta,Vt):
    A = np.diag(Delta) @ Vt
    return A
#sh
def Buildsh(T,a,b):
    sh = np.zeros(T)
    sh[0] = (a-2*b)/4
    sh[1] = b/2
    return sh

#Simulation of a n-sample of Y
def Computation_Y(T, Lambda):

    D = BuildD(T)
    
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    
    sh = Buildsh(T, a, b)
    
    
    x_tilde_true = np.zeros(T)
    for i in range(T) : # probably a more efficient way to do that
        rd = npr.uniform(0,1)
        if(rd<0.3):
            x_tilde_true[i] = npr.exponential(Lambda)

    x_true = npl.solve(D, x_tilde_true)
    Y = npr.multivariate_normal(A@x_true, np.identity(T))
    
    return Y

#Compute argmax of pi and pi_tilde distributions
def ComputeArgmax(T,Lambda, Y):
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    
    u=U@Y+sh
    x_tilde=np.sign(u)*np.maximum(np.abs(u)-Lambda,np.zeros(T))-sh
    x=npl.solve(D,x_tilde)
    return x,x_tilde

def ComputeMeans(T, Lambda, Y):
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    sh = Buildsh(T, a, b)
    mu_plus = U @ Y + Lambda*np.ones(T)
    C_plus = sps.norm.cdf(-sh-mu_plus)
    mu_minus = U @ Y - Lambda*np.ones(T)
    C_minus = 1 - sps.norm.cdf(-sh-mu_minus)
    gamma = C_plus / (C_plus + C_minus)
    mu_tilde_plus = mu_plus - np.exp(-((sh+mu_plus)**2)/2) / (np.sqrt(2*np.pi)*C_plus+1e-16)
    mu_tilde_minus = mu_minus + np.exp(-((sh+mu_minus)**2)/2) / (np.sqrt(2*np.pi)*C_minus+1e-16)
    mu_tilde = gamma*mu_tilde_plus + (1-gamma)*mu_tilde_minus
    mu = npl.solve(D, mu_tilde)
    return mu, mu_tilde

def ComputeQuantiles(T, Lambda, s, Y, niter=1e5): # Fonction de répartition et non quantiles ici !
    
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    sh = Buildsh(T, a, b)
    mu_plus = U @ Y + Lambda*np.ones(T)
    C_plus = sps.norm.cdf(-sh-mu_plus)
    mu_minus = U @ Y - Lambda*np.ones(T)
    C_minus = 1 - sps.norm.cdf(-sh-mu_minus)
    gamma = C_plus / (C_plus + C_minus)
    probas = np.zeros((int(niter),T))
    quantiles = np.zeros(T)
    q_plus = 0
    q_minus = 0
    
    for i in range(int(niter)):
        ub = -5 + i/1000
        for j in range(T):
            q_plus = sps.norm.cdf(min(ub-mu_plus[j], -sh[j]-mu_plus[j])) / C_plus[j]
            q_minus = 0
            if(ub>-sh[j]):
                q_minus = (sps.norm.cdf(ub-mu_minus[j]) + C_minus[j] - 1) / C_minus[j]
            probas[i,j] = gamma[j]*q_plus + (1 - gamma[j])*q_minus
        
    for k in range(T):
        i = 0
        while probas[i,k]<s[k]:
            i += 1
        quantiles[k] = -5 + i/1000
        
    return quantiles

def DistributionPi(x, Y, A, D, sh, Lambda):
    return np.exp((-npl.norm(Y - A@x)**2)/2-Lambda*npl.norm(D@x + sh,ord=1))

def LogDistributionPi(x, Y, A, D, sh, Lambda):
    return (-npl.norm(Y - A@x)**2)/2-Lambda*npl.norm(D@x + sh,ord=1)

def MetropolisHastings(T, Lambda, Y, niter=1e5, save=True):
    
    D = BuildD(T)
    gamma = 0.001
    gamma_final = 0.24
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    theta = np.ones(T) # maybe choose another starting point
    acceptance_cnt = 0
    sum_theta = theta
    theta_tab = np.zeros((int(niter+1), T))
    theta_tab[0,:]=theta
    theta_tilde_tab = np.zeros((int(niter+1), T))
    theta_tilde_tab[0,:]=D@theta
    burn_in = True
    theta_mean = np.zeros(T)
    cnt = 0

    # for plotting
    if save:
        gammas = [gamma]
        accepts = []
    else:
        gammas = None
        accepts = None
    
    for i in range(int(niter)):
        candidate = npr.multivariate_normal(theta, gamma*np.identity(T))
        log_alpha = LogDistributionPi(candidate, Y, A, D, sh, Lambda)-LogDistributionPi(theta, Y, A, D, sh, Lambda)
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            tmp = npr.uniform()
            if tmp <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
        # burn-in
        if burn_in and ((i+1) % 1000) == 0 : # every 1000th iteration
            gamma = gamma + (acceptance_cnt/1000 - gamma_final)*gamma
            burn_in = abs(acceptance_cnt/1000-gamma_final)>1e-2 # acceptable rate, stop adjusting gamma
            cnt=0
            # save
            if save:
                gammas.append(gamma)
                accepts.append(acceptance_cnt/1000)
            acceptance_cnt=0

        if not burn_in:
            theta_mean += theta
            cnt += 1
            
        theta_tab[i+1,:]=theta
        theta_tilde_tab[i+1,:]=D@theta

    theta_mean = theta_mean/cnt
    
    if save:
        accepts = np.array(accepts)
        gammas = np.array(gammas)
        
    return theta_tab,theta_tilde_tab, accepts, gammas, theta_mean

def MetropolisHastingsFast(T, Lambda, Y, niter=1e5, method="source"):
    """
    estimates theta and theta_tilde using the Metropolis Hastings algorithm (MH), with burn-in
    parameters:
    - T: vector space dimension, size of theta and theta_tilde
    - Lambda: parameter for the pi distribution
    - Y: parsimonious random vector of size T
    - covariance
        "source": the MH will use the identity matrix and simulate theta in the source domain
        "image": the MH will simulate theta_tilde in the image domain
    returns a tuple of two T sized vectors; means of theta and theta_tilde, obtained on all MH iterations
    """
    
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    gamma = 0.001
    gamma_final = 0.24
    acceptance_cnt = 0
    burn_in = True
    theta = np.ones(T)
    theta_mean = np.zeros(T)
    cnt = 0
    
    if method=="image":
        D_1 = npl.solve(D,np.identity(T))
        C = D_1@D_1.T
    elif method=="source":
        C = np.identity(T)
    else:
        raise Exception("method must be either 'source' or 'image'")
        
    for i in range(1,int(niter)+1):
        candidate = npr.multivariate_normal(theta, gamma*C)
        log_alpha = LogDistributionPi(candidate, Y, A, D, sh, Lambda)-LogDistributionPi(theta, Y, A, D, sh, Lambda)
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            tmp = npr.uniform()
            if tmp <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
        # burn-in
        if burn_in and ((i+1) % 1000) == 0 : # every 1000th iteration
            gamma = gamma + (acceptance_cnt/1000 - gamma_final)*gamma
            burn_in = abs(acceptance_cnt/1000-gamma_final)>1e-2 # acceptable rate, stop adjusting gamma
            acceptance_cnt=0
            cnt=0
            
        # update theta
        if not burn_in:
            theta_mean += theta
            cnt += 1
            
    theta_mean = theta_mean/cnt
    
    if method=="image":
        theta,theta_tilde = D_1@theta_mean, theta_mean
    else:
        theta,theta_tilde = theta_mean, D@theta_mean

    return theta, theta_tilde


    
#Return the quantiles q (possibly an array of quantiles) of the array sim_tab
def Quantiles(sim_tab,q,T):
    quantiles_tab=np.zeros((len(q),T))
    for i in range(len(q)):
        quantiles_tab[i,:]=np.percentile(sim_tab,q[i],axis=0)
    return quantiles_tab


