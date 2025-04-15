import scipy.stats as sps
from scipy.sparse import diags
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import csv

def PD3S(a, b, lambda_O, lambda_R):

    mu = 1e-5 #?
    
    # lambda_O, lambda_R, c = phi?, nom du fichier?
    Z = []
    with open('stuff/FranceData_Z.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            Z.append(row[0])
    phi = []
    with open('stuff/FranceData_Zphi.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            phi.append(row[0])
    c = phi #?
    T = int(len(Z)/2)

    k = [np.ones(T),-2*np.ones(T-1),np.ones(T-2)]
    offset = [0,-1,-2]
    D = diags(k,offset).toarray()/4
    C = np.diag(c)
    A = np.block([[D, np.zeros((T,T))],[np.zeros((T,T)), (lambda_O/lambda_R)*C]])
    sh = np.zeros(2*T)
    sh[0] = (a-2*b)/4
    sh[1] = b/4
    B = np.block([np.diag(phi), C])
    
    theta = np.concatenate(np.ones(T),np.zeros(T))
    s = A@theta
    M = np.zeros(2*T)
    lambda_max = np.linalg.norm(D@(D.T), ord=2)
    Lmax = np.max(lambda_max, ((lambda_O/lambda_R)**2) * np.max(c**2))
    gamma = mu/np.sqrt(Lmax)
    delta = 0.999/(gamma*Lmax)
    pi1 = (B.T)@np.linalg.inv(B@(B.T))

    for k in range(niter):
        tau = s + delta*A@theta + delta*A@M
        s = tau + delta*sh - np.max(np.abs(tau + delta*sh) - lambda_R, np.zeros(2*T))*np.sign(tau + delta*sh)
        z = theta - gamma*(A.T)@s
        u = B@z - gamma*np.diagonal(B@(B.T)) + np.sqrt((B@z - gamma*np.diagonal(B@(B.T)))**2 + 4*gamma*Z*np.diagonal(B@(B.T)))
        M = - theta
        theta = z + pi1@(0.5*u - B@z)
        M = M + theta
        
    return x,x_tilde