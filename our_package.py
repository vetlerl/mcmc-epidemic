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
    sh[1] = b/4
    return sh

#Simulation of a n-sample of Y
def Computation_Y(T, Lambda,a,b):

    param_accept = 2/T
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T,a,b)
    
    rd = npr.uniform(0, 1, T)
    x_tilde_true = np.where(rd < param_accept, npr.exponential(1/Lambda, T), 0)*(2*npr.binomial(1,1/2,T) - 1) - sh   ### 1/Lambda
    x_true = npl.solve(D, x_tilde_true)
    Y = npr.multivariate_normal(A @ x_true, np.identity(T))
    
    """
    x_tilde_true = np.zeros(T)
    for i in range(T) : # probably a more efficient way to do that
        rd = npr.uniform(0,1)
        if(rd<param_accept):
            x_tilde_true[i] = npr.exponential(1/Lambda)*(2*npr.binomial(1,1/2) - 1) - sh[i]

    x_true = npl.solve(D, x_tilde_true)
    Y = npr.multivariate_normal(A@x_true, np.identity(T))
    """
    return Y
    
def Computation_Y_debug(T, Lambda, a, b):

    param_accept = 2/T
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T,a,b)
    
    rd = npr.uniform(0, 1, T)
    x_tilde_true = np.where(rd < param_accept, npr.exponential(1/Lambda, T), 0)*(2*npr.binomial(1,1/2,T) - 1) - sh   ### 1/Lambda
    #print(2*npr.binomial(T,1/2)-1)
    #print((np.where(rd < param_accept, npr.exponential(1/Lambda, T), 0))*(2*npr.binomial(T,1/2)-1))
    x_true = npl.solve(D, x_tilde_true)
    Y = npr.multivariate_normal(A @ x_true, np.identity(T))
    
    return (Y, x_true, x_tilde_true)
    
#Create a dictionary of simulations of Y for different values of lambda parameter
def Create_DicoY(T,lambda_tab,a,b):
    npr.seed(42)
    Y_simu=dict()
    for l in lambda_tab:
        Y_simu[l]=Computation_Y(T,l,a,b)
    return Y_simu

#Compute argmax of pi and pi_tilde distributions
def ComputeArgmax(T, Lambda, Y, a, b):
    
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    
    u=U@Y+sh
    x_tilde=np.sign(u)*np.maximum(np.abs(u)-Lambda,np.zeros(T))-sh
    x=npl.solve(D,x_tilde)
    return x,x_tilde

# Stop here

def ComputeMeans(T, Lambda, Y, a, b): # without robustification here
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    sh = Buildsh(T, a, b)
    mu_plus = U @ Y + Lambda*np.ones(T)
    C_plus = sps.norm.cdf(-sh-mu_plus)
    mu_minus = U @ Y - Lambda*np.ones(T)
    C_minus = 1 - sps.norm.cdf(-sh-mu_minus)
    gamma = C_plus / (C_plus + C_minus)
    mu_tilde_plus = mu_plus - np.exp(-((sh+mu_plus)**2)/2) / (np.sqrt(2*np.pi)*C_plus)
    mu_tilde_minus = mu_minus + np.exp(-((sh+mu_minus)**2)/2) / (np.sqrt(2*np.pi)*C_minus)
    mu_tilde = gamma*mu_tilde_plus + (1-gamma)*mu_tilde_minus
    mu = npl.solve(D, mu_tilde)
    return mu, mu_tilde

def ComputeQuantiles(T, Lambda, threshold, Y, a, b, niter=int(1e5)): # Fonction de répartition et non quantiles ici !
    
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    sh = Buildsh(T, a, b)
    mu_plus = U @ Y + Lambda*np.ones(T)
    C_plus = sps.norm.cdf(-sh-mu_plus)
    mu_minus = U @ Y - Lambda*np.ones(T)
    C_minus = 1 - sps.norm.cdf(-sh-mu_minus)
    gamma = C_plus / (C_plus + C_minus)

    ub = -5 + np.arange(int(niter))/1000
    probas = np.empty((int(niter), T))

    """
    for i in range(int(niter)):
        ub = -5 + i/1000
        for j in range(T):
            q_plus = sps.norm.cdf(min(ub-mu_plus[j], -sh[j]-mu_plus[j])) / (C_plus[j]+1e-16)
            q_minus = 0
            if(ub>-sh[j]):
                q_minus = (sps.norm.cdf(ub-mu_minus[j]) + C_minus[j] - 1) / (C_minus[j] + 1e-16)
            probas[i,j] = gamma[j]*q_plus + (1 - gamma[j])*q_minus
    """
    
    for j in range(T):
        q_plus = sps.norm.cdf(np.minimum(ub[:, None]-mu_plus[j], -sh[j]-mu_plus[j])) / C_plus[j]
        q_minus = np.where(ub[:, None] > -sh[j], (sps.norm.cdf(ub[:, None]-mu_minus[j]) + C_minus[j] - 1) / C_minus[j], 0)
        probas[:, j] = gamma[j] * q_plus.squeeze() + (1 - gamma[j]) * q_minus.squeeze()

    quantiles = ub[np.argmax(probas >= threshold, axis=0)]
        
    return quantiles

#Return the quantiles q (possibly an array of quantiles) of the array sim_tab
def Quantiles(sim_tab,q,T):
    return np.percentile(sim_tab,q,axis=0)

def DistributionPi(x, Y, A, D, sh, Lambda):
    return np.exp((-npl.norm(Y - A@x)**2)/2-Lambda*npl.norm(D@x + sh,ord=1))

def LogDistributionPi(x, Y, A, D, sh, Lambda):
    return (-npl.norm(Y - A@x)**2)/2-Lambda*npl.norm(D@x + sh,ord=1)

def LogDistributionPi_Tab(x_tab, Y, A, D, sh, Lambda):
    l_tab = np.empty_like(x_tab)
    for i,xi in enumerate(x_tab):
        l_tab[i] = LogDistributionPi(xi,Y,A,D,sh,Lambda)
            #(-npl.norm(self.Y - self.A@xi)**2)/2 - self.Lambda * npl.norm(self.D@xi + self.sh,ord=1)
    return l_tab

def LogDistributionPi_Full(x, Y, A, D, sh, Lambda):
    return ((-npl.norm(Y - A@x)**2)/2, -Lambda*npl.norm(D@x + sh,ord=1))

def sub_diff(x, sh):
    return np.sign(x+sh)

def MetropolisHastingsFull(T, Lambda, Y, a,b, niter=1e5,method="source"):

    # Check the method
    is_source = method in ["source", "subdiff_source"]
    is_image = method in ["image", "subdiff_image"]
    is_subdiff = "subdiff" in method
    
    D = BuildD(T)
    gamma = 0.001
    accept_final = 0.24
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    theta = 10*np.ones(T) # maybe choose another starting point

    # Covariance matrix C
    if is_image:
        D_1 = npl.solve(D, np.identity(T))
        C = D_1@D_1.T
    elif is_source:
        C = np.identity(T)
    else:
        raise Exception("method must be either 'source' or 'image' (subdiff or not)")

    # Mean vector mu
    if is_subdiff:
        MeanProposal = CalculSubdiff  #(self, theta, gamma)
    else:
        MeanProposal = ReturnTheta

    # Proposal ratio log_alpha
    if not(is_subdiff):
        LogRatio = LogAlpha_NotSubdiff  #(self, candidate, theta, mu, gamma)
    else:
        LogRatio = LogAlpha_IsSubdiff

    """
    acceptance_cnt = 0
    sum_theta = theta
    """

    # Burn-in aux variables
    burn_in = True
    wait_conv = False
    acceptance_cnt = 0
    rd = npr.uniform(0, 1, int(niter+1))
    
    theta_tab = np.empty((int(niter+1), T))
    theta_tab[0,:]=theta
    theta_tilde_tab = np.empty((int(niter+1), T))
    theta_tilde_tab[0,:]=D@theta
    
    theta_mean = np.zeros(T)
    cnt = 0
    converge=0

    L1_tab = np.empty(int(niter+1))
    L2_tab = np.empty(int(niter+1))
    L1_tab[0], L2_tab[0] = LogDistributionPi_Full(theta_tab[0,:], Y, A, D, sh, Lambda)

    # for plotting

    cpt = 0
    gammas = np.empty(int(niter/2))
    gammas[cpt] = gamma
    accepts = np.empty(int(niter/2))
    accepts[cpt] = 0

    # burn-in loop
    for i in range(int(niter)/2):

        mu = MeanProposal(theta, gamma, A, Y, D, sh, C)
        candidate = npr.multivariate_normal(mu, gamma*C)
        log_alpha = LogRatio(candidate, theta, mu, gamma, A, Y, D, sh, C)
            
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
                    
        # burn-in
        if ((i+1) % 1000) == 0:
            accept_rate = acceptance_cnt / 1000
            if burn_in:
                gamma += (accept_rate - accept_final) * gamma
                burn_in = abs(accept_rate - accept_final) > 1e-2
                wait_conv = not burn_in
            elif wait_conv:
                converge += 1
                wait_conv = converge < 2e-4 * niter
                gamma += (accept_rate - accept_final) * gamma
                if not(wait_conv):
                    end_burn_in=i
                   
            gammas[cpt] = gamma
            accepts[cpt] = accept_rate
            cpt += 1
            acceptance_cnt = 0
        
        theta_mean += theta
        cnt += 1
        theta_tab[i+1,:]=theta
        L1_tab[i+1], L2_tab[i+1] = LogDistributionPi_Full(theta, Y, A, D, sh, Lambda)
        theta_tilde_tab[i+1,:]=D@theta

    if end_burn_in is None:
        raise ValueError("More iterations required")
    print("End of the burn-in")

    ## convergence loop
    for i in range(end_burn_in,int(niter)):
        mu = MeanProposal(theta, gamma, A, Y, D, sh, C)
        candidate = npr.multivariate_normal(mu, gamma*C)
        log_alpha = LogRatio(candidate, theta, mu, gamma, A, Y, D, sh, C)
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
            
        theta_mean += theta
        cnt += 1
        theta_tab[i+1,:] = theta
        L1_tab[i+1], L2_tab[i+1] = LogDistributionPi_Full(theta)
        theta_tilde_tab[i+1,:] = D @ theta

    theta_mean /= cnt
        
    return theta_tab,theta_tilde_tab, accepts, gammas, theta_mean, L1_tab, L2_tab,end_burn_in

def MetropolisHastings(T, Lambda, Y, a,b,niter=1e5,method="source"):

    # Check the method
    is_source = method in ["source", "subdiff_source"]
    is_image = method in ["image", "subdiff_image"]
    is_subdiff = "subdiff" in method
    
    D = BuildD(T)
    gamma = 0.001
    accept_final = 0.24
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    theta = 10*np.ones(T) # maybe choose another starting point

    # Covariance matrix C
    if is_image:
        D_1 = npl.solve(D, np.identity(T))
        C = D_1@D_1.T
    elif is_source:
        C = np.identity(T)
    else:
        raise Exception("method must be either 'source' or 'image' (subdiff or not)")

    # Mean vector mu
    if is_subdiff:
        MeanProposal = CalculSubdiff  #(self, theta, gamma)
    else:
        MeanProposal = ReturnTheta

    # Proposal ratio log_alpha
    if not(is_subdiff):
        LogRatio = LogAlpha_NotSubdiff  #(self, candidate, theta, mu, gamma)
    else:
        LogRatio = LogAlpha_IsSubdiff

    """
    acceptance_cnt = 0
    sum_theta = theta
    """

    # Burn-in aux variables
    burn_in = True
    wait_conv = False
    acceptance_cnt = 0
    rd = npr.uniform(0, 1, int(niter+1))
    
    theta_tab = np.empty((int(niter+1), T))
    theta_tab[0,:]=theta
    theta_tilde_tab = np.empty((int(niter+1), T))
    theta_tilde_tab[0,:]=D@theta
    
    theta_mean = np.zeros(T)
    cnt = 0
    converge=0

    # for plotting

    cpt = 0
    gammas = np.empty(int(niter/1000))
    gammas[cpt] = gamma
    accepts = np.empty(int(niter/1000))
    accepts[cpt] = 0

    # burn-in loop
    for i in range(int(niter/2)):

        mu = MeanProposal(theta, gamma, A, Y, D, sh, C)
        candidate = npr.multivariate_normal(mu, gamma*C)
        log_alpha = LogRatio(candidate, theta, mu, gamma, A, Y, D, sh, C, Lambda)
            
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
                    
        # burn-in
        if ((i+1) % 1000) == 0:
            accept_rate = acceptance_cnt / 1000
            if burn_in:
                gamma += (accept_rate - accept_final) * gamma
                burn_in = abs(accept_rate - accept_final) > 1e-2
                wait_conv = not burn_in
            elif wait_conv:
                converge += 1
                wait_conv = converge < 2e-4 * niter
                gamma += (accept_rate - accept_final) * gamma
                if not(wait_conv):
                    end_burn_in=i
                   
            gammas[cpt] = gamma
            accepts[cpt] = accept_rate
            cpt += 1
            acceptance_cnt = 0
        
        theta_tab[i+1,:]=theta
        theta_tilde_tab[i+1,:]=D@theta

    if(wait_conv):
        end_burn_in=int(niter/2)
    
    print("End of the burn-in")

    ## convergence loop
    for i in range(end_burn_in,int(niter)):
        mu = MeanProposal(theta, gamma, A, Y, D, sh, C)
        candidate = npr.multivariate_normal(mu, gamma*C)
        log_alpha = LogRatio(candidate, theta, mu, gamma, A, Y, D, sh, C, Lambda)
        if log_alpha >=0 :
            theta = candidate
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
            
        theta_mean += theta
        cnt += 1
        theta_tab[i+1,:] = theta
        theta_tilde_tab[i+1,:] = D @ theta

    theta_mean /= cnt
        
    return theta_tab,theta_tilde_tab, accepts, gammas, theta_mean,end_burn_in

def CalculSubdiff(theta, gamma, A, Y, D, sh, C):
    return theta - (1/2)*gamma*C@(A.T)@(Y-A@theta) - (gamma/2)*C@(D.T)@(sub_diff(D@theta, sh))

def ReturnTheta(theta, gamma, A, Y, D, sh, C):
    return theta

def LogAlpha_NotSubdiff(candidate, theta, mu, gamma, A, Y, D, sh, C, Lambda):
    log_alpha = LogDistributionPi(candidate, Y, A, D, sh, Lambda) - LogDistributionPi(theta, Y, A, D, sh, Lambda)
    return log_alpha

    # --- #
def LogAlpha_IsSubdiff(candidate, theta, mu, gamma, A, Y, D, sh, C, Lambda):
    log_alpha = LogDistributionPi(candidate, Y, A, D, sh, Lambda) - LogDistributionPi(theta, Y, A, D, sh, Lambda) 
    log_alpha -= np.log(sps.multivariate_normal.pdf(candidate, mu, gamma*C))
    log_alpha += np.log(sps.multivariate_normal.pdf(theta, CalculSubdiff(candidate, gamma, A, Y, D, sh, C), gamma*C))
    return log_alpha

def MetropolisHastingsFast(T, Lambda, Y, a,b, niter=1e5, method="source"):
    """
    estimates the mean of theta and theta_tilde using the Metropolis Hastings algorithm (MH), with burn-in
    parameters:
    - T: vector space dimension, size of theta and theta_tilde
    - Lambda: parameter for the pi distribution
    - Y: parsimonious random vector of size T
    - covariance
        "source": the MH will use the identity matrix and simulate theta in the source domain
        "image": the MH will simulate theta_tilde in the image domain
    returns a tuple of two T sized vectors; means of theta and theta_tilde, obtained on all MH iterations
    """

# Check the method
    is_source = method in ["source", "subdiff_source"]
    is_image = method in ["image", "subdiff_image"]
    is_subdiff = "subdiff" in method
    
    D = BuildD(T)
    gamma = 0.001
    accept_final = 0.24
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    theta = 10*np.ones(T) # maybe choose another starting point

    # Covariance matrix C
    if is_image:
        D_1 = npl.solve(D, np.identity(T))
        C = D_1@D_1.T
    elif is_source:
        C = np.identity(T)
    else:
        raise Exception("method must be either 'source' or 'image' (subdiff or not)")

    # Mean vector mu
    if is_subdiff:
        MeanProposal = CalculSubdiff  #(self, theta, gamma)
    else:
        MeanProposal = ReturnTheta

    # Proposal ratio log_alpha
    if not(is_subdiff):
        LogRatio = LogAlpha_NotSubdiff  #(self, candidate, theta, mu, gamma)
    else:
        LogRatio = LogAlpha_IsSubdiff

    """
    acceptance_cnt = 0
    sum_theta = theta
    """

    # Burn-in aux variables
    burn_in = True
    wait_conv = False
    acceptance_cnt = 0
    rd = npr.uniform(0, 1, int(niter+1))
    
    theta_tab = np.empty((int(niter+1), T))
    theta_tab[0,:]=theta
    theta_tilde_tab = np.empty((int(niter+1), T))
    theta_tilde_tab[0,:]=D@theta
    
    theta_mean = np.zeros(T)
    cnt = 0
    converge=0

    # for plotting

    cpt = 0
    gammas = np.empty(int(niter/2))
    gammas[cpt] = gamma
    accepts = np.empty(int(niter/2))
    accepts[cpt] = 0

    # burn-in loop
    for i in range(int(niter)/2):

        mu = MeanProposal(theta, gamma, A, Y, D, sh, C)
        candidate = npr.multivariate_normal(mu, gamma*C)
        log_alpha = LogRatio(candidate, theta, mu, gamma, A, Y, D, sh, C)
            
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
                    
        # burn-in
        if ((i+1) % 1000) == 0:
            accept_rate = acceptance_cnt / 1000
            if burn_in:
                gamma += (accept_rate - accept_final) * gamma
                burn_in = abs(accept_rate - accept_final) > 1e-2
                wait_conv = not burn_in
            elif wait_conv:
                converge += 1
                wait_conv = converge < 2e-4 * niter
                gamma += (accept_rate - accept_final) * gamma
                if not(wait_conv):
                    end_burn_in=i
                   
            if save:
                gammas[cpt] = gamma
                accepts[cpt] = accept_rate
                cpt += 1
            acceptance_cnt = 0
        
        theta_mean += theta
        cnt += 1

    if end_burn_in is None:
        raise ValueError("More iterations required")
    print("End of the burn-in")

    ## convergence loop
    for i in range(end_burn_in,int(niter)):
        mu = MeanProposal(theta, gamma, A, Y, D, sh, C)
        candidate = npr.multivariate_normal(mu, gamma*C)
        log_alpha = LogRatio(candidate, theta, mu, gamma, A, Y, D, sh, C)
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
            
        theta_mean += theta
        cnt += 1

    theta_mean /= cnt
    theta, theta_tilde = theta_mean, D@theta_mean
        
    return theta,theta_tilde




