import scipy.stats as sps
from scipy.sparse import diags
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import csv
from toy_example import BuildD

def Buildsh(T):
    sh = np.zeros(2*T)
    sh[0] = (a-2*b)/4
    sh[1] = b/4
    return sh

#Calcul de la log-densité pour une seule valeur, pour l'appliquer à un tableau , on pourra utiliser apply_along_axis
def log_pi(theta, phi,Z, lambda_R, D, sh, lambda_O,c,C):
    
    sh, _ = np.split(sh, 2)
    R, O = np.split(theta, 2) 
    log_pi_val = -np.sum(R*phi + O - Z*np.log(R*phi + c*O + 1e-10)) - lambda_R*np.linalg.norm(D@R + sh[:T], ord = 1) - lambda_O*np.linalg.norm(c@O+sh[T:], ord = 1)
    
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
    D = BuildD(T)
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

def MHRW(T, Z, phi, lambda_R,lambda_O,niter=1e5,method="source"):
    
    
    gamma = 0.001
    accept_final = 0.24
    D=BuildD(T)
    sh = Buildsh(T)
    c = np.array(phi)
    C = np.diag(c)
    theta = np.concatenate((np.ones(T),np.zeros(T))) #Starting point
    
    # Covariance matrix C
    if is_image:
        D_1 = npl.solve(D, np.identity(T))
        C = np.block([[D_1@D_1.T, np.zeros((T,T))],[np.zeros((T,T)), D_1@D_1.T]])
    elif is_source:
        C = np.identity(2*T)
    else:
        raise Exception("method must be either 'source' or 'image' ")

    # Burn-in aux variables
    end_burn_in=None
    burn_in = True
    wait_conv = False
    acceptance_cnt = 0
    rd = npr.uniform(0, 1, int(niter+1))
    
    theta_tab = np.empty((int(niter+1), 2*T))
    theta_tab[0,:]=theta
    theta_tilde_tab = np.empty((int(niter+1), 2*T))
    theta_tilde_tab[0,:]=D@theta
    converge=0

    # for plotting

    gammas = []
    accepts = []
    gammas.append(gamma)
    accepts.append(0)

    # burn-in loop
    for i in range(int(niter/2)):

        mu = theta
        candidate = npr.multivariate_normal(mu, gamma*C)
        log_alpha = log_pi(candidate, phi,Z, lambda_R, D, sh, lambda_O,c ,C )-log_pi(theta, phi,Z, lambda_R, D, sh, lambda_O,c ,C )
            
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
                
        theta_tab[i+1,:]=theta
        theta_tilde_tab[i+1,:]=D@theta    
        
        # burn-in
        if ((i+1) % 1000) == 0:
            accept_rate = acceptance_cnt / 1000
            gamma += (accept_rate - accept_final) * gamma
            gammas.append(gamma)
            accepts.append(accept_rate)
            acceptance_cnt = 0
            if burn_in:
                burn_in = abs(accept_rate - accept_final) > 1e-2
                wait_conv = not burn_in
            elif wait_conv:
                converge += 1
                wait_conv = converge < 1e-4 * niter
                if not(wait_conv):
                    end_burn_in=i
                    break
    
    if(wait_conv):
        end_burn_in=int(niter/2)
    
    print("End of the burn-in")

    ## convergence loop
    for i in range(end_burn_in,int(niter)):
        mu = theta
        candidate = npr.multivariate_normal(mu, gamma*C)
        log_alpha = log_pi(candidate, phi,Z, lambda_R, D, sh, lambda_O,c ,C )-log_pi(theta, phi,Z, lambda_R, D, sh, lambda_O,c ,C )
        if log_alpha >=0 :
            theta = candidate
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
            
        theta_tab[i+1,:] = theta
        theta_tilde_tab[i+1,:] = D @ theta
        
    return theta_tab,theta_tilde_tab, accepts, gammas,end_burn_in