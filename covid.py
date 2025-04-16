import scipy.stats as sps
from scipy.sparse import diags
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import csv

def log_pi(theta_tab, phi, c, Z, lambda_R, D, sh, lambda_O, C):
    
    niter = len(theta_tab)
    log_pi_val = np.zeros(niter)
    sh, _ = np.split(sh, 2)

    """
    print(R@phi + c@O)
    print(np.log(R*phi + c*O + 1e-10))
    print(np.sum(Z*np.log(R*phi + c*O)))
    print(lambda_R*np.linalg.norm(D@R + sh, ord = 1))
    print(lambda_O*np.linalg.norm(C@O, ord = 1))
    """
    
    for k in range(niter):
        theta = theta_tab[k,:]
        R, O = np.split(theta, 2) 
        log_pi_val[k] = -((R@phi + c@O) - np.sum(Z*np.log(R*phi + c*O + 1e-10)) + lambda_R*np.linalg.norm(D@R + sh, ord = 1) + lambda_O*np.linalg.norm(C@O, ord = 1))
        
    print(log_pi_val[:100])
    
    return log_pi_val

def PD3S(file_Z, file_phi, niter = 1e5):
    
    # lambda_O, lambda_R, c = phi?, nom du fichier?
    Z = []
    with open(file_Z) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            Z.append(float(row[0]))
    phi = []
    with open(file_phi) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            phi.append(float(row[0]))

    a = 0.73597
    b = 0.73227
    #Z = Z[2:]
    #phi = phi[2:]

    T = len(Z)
    c = np.array(phi) #?
    Z = np.array(Z)
    
    lambda_O = 0.05
    lambda_R = 3.5*np.std(Z)
    k = [np.ones(T),-2*np.ones(T-1),np.ones(T-2)]
    offset = [0,-1,-2]
    D = diags(k,offset).toarray()/4
    C = np.diag(c)
    A = np.block([[D, np.zeros((T,T))],[np.zeros((T,T)), (lambda_O/lambda_R)*C]])
    sh = np.zeros(2*T)
    sh[0] = (a-2*b)/4
    sh[1] = b/4
    B = np.block([np.diag(phi), C])

    mu = 1e-5 #?
    theta = np.concatenate((np.ones(T),np.zeros(T)))
    s = A@theta
    M = np.zeros(2*T)
    lambda_max = np.linalg.norm(D@(D.T), ord=2)
    Lmax = max(lambda_max, ((lambda_O/lambda_R)**2) * np.max(c**2))
    gamma = mu/np.sqrt(Lmax)
    delta = 0.999/(gamma*Lmax)
    pi1 = (B.T)@np.linalg.inv(B@(B.T))

    niter = int(niter)
    theta_tab = np.zeros((niter+1, 2*T))
    theta_tab[0,:] = theta

    for k in range(niter):
        tau = s + delta*A@theta + delta*A@M
        s = tau + delta*sh - np.maximum(np.abs(tau + delta*sh) - lambda_R, np.zeros(2*T))*np.sign(tau + delta*sh)
        z = theta - gamma*(A.T)@s
        u = B@z - gamma*np.diagonal(B@(B.T)) + np.sqrt((B@z - gamma*np.diagonal(B@(B.T)))**2 + 4*gamma*Z*np.diagonal(B@(B.T)))
        M = - theta
        theta = z + pi1@(0.5*u - B@z)
        M = M + theta
        theta_tab[k+1,:] = theta
        
    return theta

