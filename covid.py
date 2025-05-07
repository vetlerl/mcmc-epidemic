import scipy.stats as sps
from scipy.sparse import diags
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as plt
import csv
from scipy.stats import multivariate_normal
from numpy.linalg import slogdet, inv

def BuildD(T):
    k = [np.ones(T),-2*np.ones(T-1),np.ones(T-2)]
    offset = [0,-1,-2]
    D = diags(k,offset).toarray()/4
    return D
    
def Buildbarsh(T,a,b):
    barsh = np.zeros(2*T)
    barsh[0] = (a-2*b)/4
    barsh[1] = b/4
    return barsh

#Calcul de la log-densité pour une seule valeur, pour l'appliquer à un tableau , on pourra utiliser apply_along_axis
def log_pi(theta, phi,Z, lambda_R, D, barsh, lambda_O,c,C):
    
    shR, shO = np.split(barsh, 2)
    R, O = np.split(theta, 2) 
    temp_c = R*phi + c*O
    log_pi_val = -np.sum(temp_c - Z*np.log(temp_c)) - lambda_R*np.linalg.norm(D@R + shR, ord = 1) - lambda_O*np.linalg.norm(C@O+shO, ord = 1)
    
    return log_pi_val

def log_pi_with_fixed_O(theta_R, MAPO,phi,Z, lambda_R, D, barsh, lambda_O,c,C):
    theta_complete = np.concatenate([theta_R, MAPO])
    return log_pi(theta_complete, phi, Z, lambda_R, D, barsh, lambda_O, c, C)

def log_pi_with_fixed_R(theta_O, MAPR,phi,Z, lambda_R, D, barsh, lambda_O,c,C):
    theta_complete = np.concatenate([MAPR, theta_O])
    return log_pi(theta_complete, phi, Z, lambda_R, D, barsh, lambda_O, c, C)
    
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

def CalculSubdiff(Z,phi,theta,A,c,lambda_R,lambda_O,barsh,gamma,Cov):
    
    R,O=np.split(theta,2)
    gradfR=phi-Z*phi/(R*phi+c*O)
    gradfO=c-Z*c/(R*phi+c*O)
    subgradg=lambda_R*(A.T)@np.sign(A@theta+barsh)
    gradf=np.concatenate((gradfR,gradfO))
    return theta - (gamma/2)*Cov@gradf - (gamma/2)*Cov@subgradg

def DriftImage(Z,phi,theta,theta_tilde,gamma,lambda_R,c,barsh):
    T =len(Z)
    R,O=np.split(theta,2)
    gradfR=phi-Z*phi/(R*phi+c*O)
    gradfO=c-Z*c/(R*phi+c*O)
    gradf=np.concatenate((gradfR,gradfO))
    tau = theta_tilde - (gamma/2)*gradf
    return -barsh + np.sign(tau + barsh)*np.maximum(np.abs(tau + barsh) - lambda_R*(gamma/2), np.zeros(2*T))
    
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

def MHRW(T, Z, phi, lambda_R,lambda_O,niter=1e5,method="source"):
    
    a = 0.73597
    b = 0.73227
    gamma = 0.001
    accept_final = 0.24
    D=BuildD(T)
    barsh = Buildbarsh(T,a,b)
    c = np.array(phi)
    C = np.diag(c)
    theta = np.concatenate((np.ones(T), np.zeros(T)))
    A = np.block([[D, np.zeros((T,T))],[np.zeros((T,T)), (lambda_O/lambda_R)*C]])
    is_image = method in ["image"]
    is_source = method in ["source"]
    
    # Covariance matrix Cov
    if is_image:
        A_1 = npl.solve(A, np.identity(2*T))
        #Cov = A_1@A_1.T
        linear_combination = A_1
    elif is_source:
        Cov = np.identity(2*T)
        linear_combination = np.identity(2*T)
    else:
        raise Exception("method must be either 'source' or 'image' ")

    # Burn-in aux variables
    end_burn_in=int(niter/2)
    burn_in = True
    wait_conv = False
    acceptance_cnt = 0
    rd = npr.uniform(0, 1, int(niter+1))
    theta_tab = np.empty((int(niter+1), 2*T))
    theta_tab[0,:]=theta
    #theta_tilde_tab = np.empty((int(niter+1), 2*T))
    converge=0
    logpi_courant = log_pi(theta, phi,Z, lambda_R, D, barsh, lambda_O,c ,C)

    # for plotting

    gammas = []
    accepts = []
    gammas.append(gamma)
    accepts.append(0)
    
    print(linear_combination)

    # burn-in loop
    for i in range(int(niter/2)):

        mu = theta
        candidate = mu + np.sqrt(gamma)*linear_combination@npr.normal(0, 1, 2*T)
        #Verify values of R>0 and R+O>0
        R,O=np.split(candidate,2)
        if(np.all(R>=0))and(np.all((R*phi+c*O)>0)):
            logpi_candidate = log_pi(candidate, phi,Z, lambda_R, D, barsh, lambda_O,c ,C)
            log_alpha = logpi_candidate - logpi_courant
            
            if log_alpha >=0 :
                theta = candidate
                acceptance_cnt += 1
                logpi_courant = logpi_candidate
            else:
                if rd[i] <= np.exp(log_alpha): # probability alpha of success
                    theta = candidate
                    acceptance_cnt += 1
                    logpi_courant = logpi_candidate

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
            print(i)
            
    if(wait_conv):
        end_burn_in=int(niter/2)
    
    print("End of the burn-in")

    ## convergence loop
    for i in range(end_burn_in,int(niter)):
        mu = theta
        candidate = mu + np.sqrt(gamma)*linear_combination@npr.normal(0, 1, 2*T)
        #Verify values of R>0 and R+O>0
        R,O=np.split(candidate,2)
        if(np.all(R>=0))and(np.all((R*phi+c*O)>0)):
            logpi_candidate = log_pi(candidate, phi,Z, lambda_R, D, barsh, lambda_O,c ,C )
            
            log_alpha = logpi_candidate -logpi_courant
            
            if log_alpha >=0 :
                theta = candidate
                logpi_courant = logpi_candidate
            else:
                if rd[i] <= np.exp(log_alpha): # probability alpha of success
                    theta = candidate
                    logpi_courant = logpi_candidate
                
        theta_tab[i+1,:]=theta 
        if ((i+1) % 50000) == 0:
            print(i)   
            
    #theta_tilde_tab = np.einsum('ij,kj->ki', A, theta_tab)
    return theta_tab, accepts, gammas,end_burn_in

def fast_logpdf(theta, mu, gamma, eigvals, eigvecs, dim):
    scaled_eigvals = gamma * eigvals
    inv_cov = eigvecs @ np.diag(1.0 / scaled_eigvals) @ eigvecs.T
    delta = theta - mu
    return -0.5 * (np.dot(delta, inv_cov @ delta) )

def MHSubdiff(T, Z, phi, lambda_R,lambda_O,niter=1e5,method="source"):
    
    a = 0.73597
    b = 0.73227
    gamma = 0.001
    accept_final = 0.24
    D=BuildD(T)
    barsh = Buildbarsh(T,a,b)
    c = np.array(phi)
    C = np.diag(c)
    theta = np.concatenate((np.ones(T), np.zeros(T)))
    A = np.block([[D, np.zeros((T,T))],[np.zeros((T,T)), (lambda_O/lambda_R)*C]])
    is_image = method in ["image"]
    is_source = method in ["source"]
    
    # Covariance matrix Cov
    if is_image:
        A_1 = npl.solve(A, np.identity(2*T))
        Cov = A_1@A_1.T
        linear_combination = A_1
    elif is_source:
        Cov = np.identity(2*T)
        linear_combination = np.identity(2*T)
    else:
        raise Exception("method must be either 'source' or 'image' ")
    dim = Cov.shape[0]
    eigvals, eigvecs = np.linalg.eigh(Cov)  # symmetric matrix, so eigh is faster and more stable

    # Burn-in aux variables
    end_burn_in=int(niter/2)
    burn_in = True
    wait_conv = False
    acceptance_cnt = 0
    rd = npr.uniform(0, 1, int(niter+1))
    theta_tab = np.empty((int(niter+1), 2*T))
    theta_tab[0,:]=theta
    #theta_tilde_tab = np.empty((int(niter+1), 2*T))
    converge=0
    logpi_courant = log_pi(theta, phi,Z, lambda_R, D, barsh, lambda_O,c ,C)

    # for plotting

    gammas = []
    accepts = []
    gammas.append(gamma)
    accepts.append(0)
    
    # burn-in loop
    for i in range(int(niter/2)):

        mu = CalculSubdiff(Z,phi,theta,A,c,lambda_R,lambda_O,barsh,gamma,Cov)
        candidate = mu + np.sqrt(gamma)*linear_combination@npr.normal(0, 1, 2*T)
        #Verify values of R>0 and R+O>0
        R,O=np.split(candidate,2)
        if(np.all(R>=0))and(np.all((R*phi+c*O)>0)):
        
            logpi_candidate = log_pi(candidate, phi,Z, lambda_R, D, barsh, lambda_O,c ,C )        
            dens_courant= fast_logpdf(theta, CalculSubdiff(Z,phi,candidate,A,c,lambda_R,lambda_O,barsh,gamma,Cov), gamma, eigvals, eigvecs, dim)
            dens_candidate=fast_logpdf(candidate, mu, gamma, eigvals, eigvecs, dim)
            log_alpha = logpi_candidate - logpi_courant + dens_courant - dens_candidate
    
            if log_alpha >=0 :
                theta = candidate
                acceptance_cnt += 1
                logpi_courant = logpi_candidate
            else:
                if rd[i] <= np.exp(log_alpha): # probability alpha of success
                    theta = candidate
                    acceptance_cnt += 1
                    logpi_courant = logpi_candidate

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
            print(i+1)
            
    if(wait_conv):
        end_burn_in=int(niter/2)
    
    print("End of the burn-in")

    ## convergence loop
    for i in range(end_burn_in,int(niter)):
        mu = CalculSubdiff(Z,phi,theta,A,c,lambda_R,lambda_O,barsh,gamma,Cov)
        candidate = mu + np.sqrt(gamma)*linear_combination@npr.normal(0, 1, 2*T)
        #Verify values of R>0 and R+O>0
        R,O=np.split(candidate,2)
        if(np.all(R>=0))and(np.all((R*phi+c*O)>0)):
            
            logpi_candidate = log_pi(candidate, phi,Z, lambda_R, D, barsh, lambda_O,c ,C )
            dens_courant= fast_logpdf(theta, CalculSubdiff(Z,phi,candidate,A,c,lambda_R,lambda_O,barsh,gamma,Cov), gamma, eigvals, eigvecs, dim)
            dens_candidate= fast_logpdf(candidate, mu, gamma, eigvals, eigvecs, dim)
            log_alpha = logpi_candidate - logpi_courant-dens_candidate+dens_courant
    
            if log_alpha >=0 :
                theta = candidate
                logpi_courant = logpi_candidate
            else:
                if rd[i] <= np.exp(log_alpha): # probability alpha of success
                    theta = candidate
                    logpi_courant = logpi_candidate
                
        theta_tab[i+1,:]=theta 
        if ((i+1) % 50000) == 0:
            print(i)  
            
    #theta_tilde_tab = np.einsum('ij,kj->ki', A, theta_tab)
    return theta_tab, accepts, gammas,end_burn_in

def MHProxImage(T, Z, phi, lambda_R,lambda_O,niter=1e5):
    
    a = 0.73597
    b = 0.73227
    gamma = 0.001
    accept_final = 0.24
    D=BuildD(T)
    barsh = Buildbarsh(T,a,b)
    c = np.array(phi)
    C = np.diag(c)
    theta =np.concatenate((np.ones(T), np.zeros(T)))
    #theta_tilde = np.concatenate((1e-4*np.ones(T), np.zeros(T)))
    A = np.block([[D, np.zeros((T,T))],[np.zeros((T,T)), (lambda_O/lambda_R)*C]])
    theta_tilde = A@theta
    A_1=np.linalg.solve(A,np.identity(2*T))
    #{theta=A_1@theta_tilde
    #is_image = method in ["image"]
    #is_source = method in ["source"]
    logpi_courant = log_pi(theta, phi,Z, lambda_R, D, barsh, lambda_O,c ,C)
    
    Cov = np.identity(2*T)

    # Burn-in aux variables
    end_burn_in=int(niter/2)
    burn_in = True
    wait_conv = False
    acceptance_cnt = 0
    rd = npr.uniform(0, 1, int(niter+1))
    theta_tab = np.empty((int(niter+1), 2*T))
    theta_tab[0,:]=theta
    #theta_tilde_tab = np.empty((int(niter+1), 2*T))
    converge=0

    # for plotting

    gammas = []
    accepts = []
    gammas.append(gamma)
    accepts.append(0)

    dim = Cov.shape[0]
    eigvals, eigvecs = np.linalg.eigh(Cov)  # symmetric matrix, so eigh is faster and more stable

    
    # burn-in loop
    for i in range(int(niter/2)):

        mu_tilde = DriftImage(Z,phi,theta,theta_tilde,gamma,lambda_R,c,barsh)
        candidate_tilde = mu_tilde + np.sqrt(gamma)*npr.normal(0, 1, 2*T)
        candidate=A_1@candidate_tilde
        #Verify values of R>0 and R+O>0
        R,O=np.split(candidate,2)
        
        if(np.all(R>=0))and(np.all((R*phi+c*O)>0)):
            logpi_candidate = log_pi(candidate, phi,Z, lambda_R, D, barsh, lambda_O,c ,C )
            dens_courant= fast_logpdf(theta_tilde, DriftImage(Z,phi,candidate,candidate_tilde,gamma,lambda_R,c,barsh), gamma, eigvals, eigvecs, dim)
            dens_candidate=fast_logpdf(candidate_tilde, mu_tilde, gamma, eigvals, eigvecs, dim)
            log_alpha = logpi_candidate - logpi_courant-dens_candidate+dens_courant
                
            if log_alpha >=0 :
                theta_tilde = candidate_tilde
                theta = candidate
                acceptance_cnt += 1
                logpi_courant = logpi_candidate
            else:
                if rd[i] <= np.exp(log_alpha): # probability alpha of success
                    theta_tilde = candidate_tilde
                    theta = candidate
                    acceptance_cnt += 1
                    logpi_courant = logpi_candidate
                
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
            print(i)
    if(wait_conv):
        end_burn_in=int(niter/2)
    
    print("End of the burn-in")

    ## convergence loop
    for i in range(end_burn_in,int(niter)):
        mu_tilde = DriftImage(Z,phi,theta,theta_tilde,gamma,lambda_R,c,barsh)
        candidate_tilde = mu_tilde + np.sqrt(gamma)*npr.normal(0, 1, 2*T)
        candidate=A_1@candidate_tilde
        #Verify values of R>0 and R+O>0
        R,O=np.split(candidate,2)
        if(np.all(R>=0))and(np.all((R*phi+c*O)>0)):
            logpi_candidate = log_pi(candidate, phi,Z, lambda_R, D, barsh, lambda_O,c ,C )
            dens_courant= fast_logpdf(theta_tilde, DriftImage(Z,phi,candidate,candidate_tilde,gamma,lambda_R,c,barsh), gamma, eigvals, eigvecs, dim)
            dens_candidate=fast_logpdf(candidate_tilde, mu_tilde, gamma, eigvals, eigvecs, dim)
            log_alpha = logpi_candidate - logpi_courant-dens_candidate+dens_courant
                
            if log_alpha >=0 :
                theta_tilde = candidate_tilde
                theta = candidate
                acceptance_cnt += 1
                logpi_courant = logpi_candidate
            else:
                if rd[i] <= np.exp(log_alpha): # probability alpha of success
                    theta_tilde = candidate_tilde
                    theta = candidate
                    acceptance_cnt += 1
                    logpi_courant = logpi_candidate
                
        theta_tab[i+1,:]=theta     
        if ((i+1) % 50000) == 0:
            print(i)    
            
    #theta_tab = np.einsum('ij,kj->ki', A_1, theta_tab)
    return theta_tab, accepts, gammas,end_burn_in