import scipy.stats as sps
from scipy.sparse import diags
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import csv
from toy_example import BuildD

def Buildbarsh(T,a,b):
    barsh = np.zeros(2*T)
    barsh[0] = (a-2*b)/4
    barsh[1] = b/4
    return barsh

#Calcul de la log-densité pour une seule valeur, pour l'appliquer à un tableau , on pourra utiliser apply_along_axis
def log_pi(theta, phi,Z, lambda_R, D, barsh, lambda_O,c,C):
    
    shR, shO = np.split(barsh, 2)
    R, O = np.split(theta, 2) 
    log_pi_val = -np.sum(R*phi + O - Z*np.log(R*phi + c*O + 1e-10)) - lambda_R*np.linalg.norm(D@R + shR, ord = 1) - lambda_O*np.linalg.norm(C@O+shO, ord = 1)
    
    return log_pi_val


def load_data(file_Z,file_phi):
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
    return np.array(Z),np.array(phi)


def PD3S(Z, phi, niter = 1e5):
    
    # lambda_O, lambda_R, c = phi?, nom du fichier?
    
    a = 0.73597
    b = 0.73227
    #Z = Z[2:]
    #phi = phi[2:]

    T = len(Z)
    c = phi #?
    
    lambda_O = 0.05
    lambda_R = 3.5*np.std(Z)
    D = BuildD(T)
    C = np.diag(c)
    A = np.block([[D, np.zeros((T,T))],[np.zeros((T,T)), (lambda_O/lambda_R)*C]])
    barsh = Buildbarsh(T,a,b)
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
        s = tau + delta*barsh - np.maximum(np.abs(tau + delta*barsh) - lambda_R, np.zeros(2*T))*np.sign(tau + delta*barsh)
        z = theta - gamma*(A.T)@s
        u = B@z - gamma*np.diagonal(B@(B.T)) + np.sqrt((B@z - gamma*np.diagonal(B@(B.T)))**2 + 4*gamma*Z*np.diagonal(B@(B.T)))
        M = - theta
        theta = z + pi1@(0.5*u - B@z)
        M = M + theta
        theta_tab[k+1,:] = theta
        
    return theta

def MHRW(T, Z, phi, lambda_R,lambda_O,MAP,niter=1e5,method="source"):
    
    a = 0.73597
    b = 0.73227
    gamma = 0.001
    accept_final = 0.24
    D=BuildD(T)
    barsh = Buildbarsh(T,a,b)
    c = np.array(phi)
    C = np.diag(c)
    theta = np.concatenate((np.ones(T),np.zeros(T))) #Starting point
    A = np.block([[D, np.zeros((T,T))],[np.zeros((T,T)), (lambda_O/lambda_R)*C]])
    is_image = method in ["image", "subdiff_image"]
    is_source = method in ["source", "subdiff_source"]
    
    # Covariance matrix Cov
    if is_image:
        A_1 = npl.solve(A, np.identity(2*T))
        Cov = A_1@A_1.T
    elif is_source:
        Cov = np.identity(2*T)
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
    converge=0

    # for plotting

    gammas = []
    accepts = []
    gammas.append(gamma)
    accepts.append(0)
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Évolution de R")
    ax2.set_title("Évolution de O")
    
    #Computation of the MAP for comparison
    MAPR,MAPO=np.split(MAP,2)
    
    # burn-in loop
    for i in range(int(niter/2)):

        mu = theta
        candidate = npr.multivariate_normal(mu, gamma*Cov)
        log_alpha = log_pi(candidate, phi,Z, lambda_R, D, barsh, lambda_O,c ,C )-log_pi(theta, phi,Z, lambda_R, D, barsh, lambda_O,c ,C )
            
        if log_alpha >=0 :
            theta = candidate
            acceptance_cnt += 1
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
                acceptance_cnt += 1
                
        theta_tab[i+1,:]=theta   
        
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
                wait_conv = converge < 2e-4 * niter
                if not(wait_conv):
                    end_burn_in=i
                    break
        if ((i+1) % 50000) == 0:
            R,O=np.split(theta,2)
            ax1.clear()
            ax2.clear()
            ax1.plot(R,label=f"Simulation of R at iter {i}")
            ax1.plot(MAPR,label="MAP R")
            ax2.plot(O,label=f"Simulation of O at iter {i}")
            ax2.plot(MAPR,label="MAP O")
            ax1.set_title("Évolution de R")
            ax2.set_title("Évolution de O")
            ax1.legend()
            ax2.legend()
        
            plt.pause(0.1) 
    if(wait_conv):
        end_burn_in=int(niter/2)
    
    print("End of the burn-in")

    ## convergence loop
    for i in range(end_burn_in,int(niter)):
        mu = theta
        candidate = npr.multivariate_normal(mu, gamma*Cov)
        log_alpha = log_pi(candidate, phi,Z, lambda_R, D, barsh, lambda_O,c ,C )-log_pi(theta, phi,Z, lambda_R, D, barsh, lambda_O,c ,C )
        if log_alpha >=0 :
            theta = candidate
        else:
            if rd[i] <= np.exp(log_alpha): # probability alpha of success
                theta = candidate
            
        theta_tab[i+1,:] = theta
        if ((i+1) % 50000) == 0:
            R,O=np.split(theta,2)
            ax1.clear()
            ax2.clear()
            ax1.plot(R,label=f"Simulation of R at iter {i}")
            ax1.plot(MAPR,label="MAP R")
            ax2.plot(O,label=f"Simulation of O at iter {i}")
            ax2.plot(MAPO,label="MAP O")
            ax1.legend()
            ax2.legend()
        
            plt.pause(0.1) 
            
    plt.ioff()
    plt.show()
    theta_tilde_tab = np.einsum('ij,kj->ki', A, theta_tab)
    return theta_tab,theta_tilde_tab, accepts, gammas,end_burn_in