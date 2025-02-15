# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:17:19 2025

@author: arman paul et vetle
"""

import scipy.stats
from scipy.sparse import diags
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

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
        rd = np.random.uniform(0,1)
        if(rd<0.3):
            x_tilde_true[i] = np.random.exponential(Lambda)

    x_true = npl.solve(D, x_tilde_true)
    Y = np.random.multivariate_normal(A@x_true, np.identity(T))
    
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
    #assert ((0 <= -sh-mu_plus) & (-sh-mu_plus <= 1)).all()
    C_plus = scipy.stats.norm.cdf(-sh-mu_plus)
    mu_minus = U @ Y - Lambda*np.ones(T)
    #assert ((0 <= -sh-mu_minus) & (-sh-mu_minus <= 1)).all()
    C_minus = 1 - scipy.stats.norm.cdf(-sh-mu_minus)
    gamma = C_plus / (C_plus + C_minus)
    mu_tilde_plus = mu_plus - np.exp(-((sh+mu_plus)**2)/2) / (np.sqrt(2*np.pi)*C_plus)
    mu_tilde_minus = mu_minus + np.exp(-((sh+mu_minus)**2)/2) / (np.sqrt(2*np.pi)*C_minus)
    mu_tilde = gamma*mu_tilde_plus + (1-gamma)*mu_tilde_minus
    mu = npl.solve(D, mu_tilde)
    return mu, mu_tilde

def ComputeQuantiles(T, Lambda, s, Y): # Fonction de répartition et non quantiles ici !
    
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    sh = Buildsh(T, a, b)
    mu_plus = U @ Y + Lambda*np.ones(T)
    C_plus = scipy.stats.norm.cdf(-sh-mu_plus)
    mu_minus = U @ Y - Lambda*np.ones(T)
    C_minus = 1 - scipy.stats.norm.cdf(-sh-mu_minus)
    gamma = C_plus / (C_plus + C_minus)
    probas = np.zeros((10000,T))
    quantiles = np.zeros(T)
    q_plus = 0
    q_minus = 0
    
    for i in range(10000):
        ub = -5 + i/1000
        for j in range(T):
            q_plus = scipy.stats.norm.cdf(min(ub-mu_plus[j], -sh[j]-mu_plus[j])) / C_plus[j]
            q_minus = 0
            if(ub>-sh[j]):
                q_minus = (scipy.stats.norm.cdf(ub-mu_minus[j]) + C_minus[j] - 1) / C_minus[j]
            probas[i,j] = gamma[j]*q_plus + (1 - gamma[j])*q_minus
        
    for k in range(T):
        i = 0
        while(probas[i,k]<s[k]):
            i += 1
        quantiles[k] = -5 + i/1000
        
    return quantiles

def DistributionPi(x, Y, A, D, sh, Lambda):
    return np.exp((-np.linalg.norm(Y - A@x)**2)/2-Lambda*np.linalg.norm(D@x + sh,ord=1))

def LogDistributionPi(x, Y, A, D, sh, Lambda):
    return (-np.linalg.norm(Y - A@x)**2)/2-Lambda*np.linalg.norm(D@x + sh,ord=1)

def MetropolisHastings(T, Lambda, Y, niter=1e6, a=1, b=2):
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
    theta_tab[0,:]=D@theta
    
    for i in range(int(niter)):
        candidate = np.random.multivariate_normal(theta, gamma*np.identity(T))
        log_alpha = LogDistributionPi(candidate, Y, A, D, sh, Lambda)-LogDistributionPi(theta, Y, A, D, sh, Lambda)
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            tmp = np.random.uniform()
            if tmp <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
        # burn-in
        if ((i+1) % 1000) == 0 : # every 1000th iteration
            gamma = gamma + (acceptance_cnt/1000 - gamma_final)*gamma
            acceptance_cnt=0
            
        sum_theta = np.add(sum_theta, theta)
        theta_tab[i+1,:]=D@theta
        
    return (1/niter)*(sum_theta),theta_tab

#Return the quantiles q (possibly an array of quantiles) of the array sim_tab
def Quantiles(sim_tab,q,T):
    quantiles_tab=np.zeros((len(q),T))
    for i in range(len(q)):
        quantiles_tab[i,:]=np.percentile(sim_tab,q[i],axis=0)
    return quantiles_tab
    
#Test of the different functions 
T = 5
Lambda = 1

D = BuildD(T)
U, Delta, Vt = BuildUVDelta(D)
A = BuildA(Delta, Vt)
sh = Buildsh(T, a, b)
Y = Computation_Y(T, Lambda)
x,x_tilde = ComputeArgmax(T,Lambda, Y)
mu,mu_tilde = ComputeMeans(T,Lambda, Y)
q1 = ComputeQuantiles(T,Lambda,0.975*np.ones(T), Y)
q2 = ComputeQuantiles(T,Lambda,0.025*np.ones(T), Y)
med = ComputeQuantiles(T,Lambda,0.5*np.ones(T), Y)
Mean,sim_tab = MetropolisHastings(T,Lambda, Y)
q = np.array([2.5,50,97.5])
quantiles_emp = Quantiles(sim_tab, q,T)
"""
print(f"We test our functions for T={T} and Lambda={Lambda}")
print("------- variables -------")
print(f"D = {D}")
print(f"A = {A}")
print(f"sh = {sh}")
print("------- functions -------")
print(" * ComputeArgmax: ")
print(f"x_tilde = {x_tilde}")
print(f"x = {x}")
"""
print(" * ComputeMeans: ")
print(f"mu = {mu}")
print(f"mu_tilde = {mu_tilde}")

print(" * MetropolisHastings")
print(f"Moy empirique = {Mean}")
print(f"Moy théorique = {mu}")

print(" * ComputeQuantiles:")
print(f"97.5% quantile = {q1}")
print(f"Median = {med}")
print(f"2.5% quantile = {q2}")

print(f"Quantile empirique 97.5% = {quantiles_emp[2]}")
print(f"Mediane empirique = {quantiles_emp[1]}")
print(f"Quantile empirique 2.5% = {quantiles_emp[0]}")


#Plot of theoritical results
plt.figure()
plt.plot(mu_tilde,color="blue",label="Mean")
plt.plot(x_tilde,color="darksalmon",label="Argmax")
plt.plot(med,'g--',label="Median")
plt.plot(q1,'b--',label="97.5 quantile")
plt.plot(q2,'r--',label="2.5 quantile")
plt.title("Theoritical results for pi tilde")
plt.legend()
plt.show()