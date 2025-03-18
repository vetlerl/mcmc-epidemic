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


### Attention vos notations ne sont pas toujours uniformes

### Je retire plusieurs boucles for

### Je remplace tous vos   for i in range(len(***))  par   for i,_ in enumerate(***)
# --> plus rapide

### Je remplace plusieurs de vos  np.zeroes  par des  np.empty
# --> meilleure gestion de la mémoire


class ToyModel:
    def __init__(self, T, a=1, b=2):
        self.a = a
        self.b = b
        self.T = T

        # build D
        k = [np.ones(self.T), -2*np.ones(self.T-1), np.ones(self.T-2)]
        offset = [0,-1,-2]
        self.D = diags(k,offset).toarray()/4
        
        # build U, Delta, transpose-V
        self.U, self.Delta, self.Vt = npl.svd(self.D, full_matrices=False)

        # build A
        self.A = np.diag(self.Delta) @ self.Vt
        
        # build shift
        self.sh = np.zeros(self.T)
        self.sh[0] = (self.a-2*self.b)/4
        self.sh[1] = self.b/2


    # --------------- #
    # SIMULATION of Y #
    # --------------- #
    def Computation_Y(self, Lambda):
        #Simulation of a n-sample of Y
        rd = npr.uniform(0, 1, self.T)
        x_tilde_true = np.where(rd < 0.3, npr.exponential(1/Lambda, self.T), 0)   ### 1/Lambda
        x_true = npl.solve(self.D, x_tilde_true)
        Y = npr.multivariate_normal(self.A @ x_true, np.identity(self.T))
        return Y

    # --- #
    def CreateDico_Y(self, Lambda_tab):
        #Create a dictionnary of simulation of Y for different values of Lambda
        npr.seed(42)
        Y_simu = dict()
        for l in Lambda_tab:
            Y_simu[l] = self.Computation_Y(l)
        return Y_simu

    
    # ------------- # ------------- #
    # Mean, Quantiles, Argmax, etc. #
    # ------------- # ------------- #
    def ComputeArgmax(self, Lambda, Y):
        #Compute argmax of pi and pi_tilde distributions
        u = self.U @ Y + self.sh
        x_tilde = np.sign(u) * np.maximum(np.abs(u)-Lambda,np.zeros(self.T)) - self.sh
        x = npl.solve(self.D, x_tilde)
        return x, x_tilde

    def MeanQuantilesInit(self, Lambda, Y):
        mu_plus = self.U @ Y + Lambda * np.ones(self.T)
        C_plus = sps.norm.cdf(-self.sh-mu_plus)
        mu_minus = self.U @ Y - Lambda * np.ones(self.T)
        C_minus = 1 - sps.norm.cdf(-self.sh-mu_minus)
        gamma = C_plus / (C_plus + C_minus)
        return mu_plus, mu_minus, C_plus, C_minus, gamma
    
    # --- #
    def ComputeMeans(self, Lambda, Y):
        mu_plus, mu_minus, C_plus, C_minus, gamma = self.MeanQuantilesInit(Lambda, Y)
        mu_tilde_plus = mu_plus - np.exp(-((self.sh+mu_plus)**2)/2) / (np.sqrt(2*np.pi)*C_plus+1e-16)
        mu_tilde_minus = mu_minus + np.exp(-((self.sh+mu_minus)**2)/2) / (np.sqrt(2*np.pi)*C_minus+1e-16)
        mu_tilde = gamma*mu_tilde_plus + (1-gamma)*mu_tilde_minus
        mu = npl.solve(self.D, mu_tilde)
        return mu, mu_tilde
        
    # --- #
    def ComputeQuantiles(self, Lambda, Y, threshold, niter=int(1e5)):
        # Fonction de répartition et non quantiles ici !
        mu_plus, mu_minus, C_plus, C_minus, gamma = self.MeanQuantilesInit(Lambda, Y)
        
        ub = -5 + np.arange(int(niter)) / 1000
        probas = np.empty((int(niter), self.T))
        
        if np.isscalar(threshold):
            threshold = np.full(self.T, threshold)
    
        for j in range(self.T):
            q_plus = sps.norm.cdf(np.minimum(ub[:, None]-mu_plus[j], -self.sh[j]-mu_plus[j])) / C_plus[j]
            q_minus = np.where(ub[:, None] > -self.sh[j],
                                    (sps.norm.cdf(ub[:, None]-mu_minus[j]) + C_minus[j] - 1) / C_minus[j], 0)
            probas[:, j] = gamma[j] * q_plus.squeeze() + (1 - gamma[j]) * q_minus.squeeze()
        
        quantiles = ub[np.argmax(probas >= threshold, axis=0)]
        return quantiles

    # --- #
    def Quantiles(self, sim_tab, q):
        #Return the quantiles q (possibly an array of quantiles) of the array sim_tab
        return np.percentile(sim_tab, q, axis=0)


    # ---------------- # --------------- #
    # Auxiliary functions for simulation #
    # ---------------- # --------------- #    
    # Here, assume that self.Y and self.Lambda are well defined --> cf HMInit

    # --- #
    def DistributionPi(self, x):
        return np.exp( (-npl.norm(self.Y - self.A@x)**2)/2 - self.Lambda * npl.norm(self.D@x + self.sh,ord=1) )

    # --- #
    def LogDistributionPi(self, x):
        return (-npl.norm(self.Y - self.A@x)**2)/2 - self.Lambda * npl.norm(self.D@x + self.sh,ord=1)

    # --- #
    def LogDistributionPi_Tab(self, x_tab):
        l_tab = np.empty_like(x_tab)
        for i,xi in enumerate(x_tab):
            l_tab[i] = self.LogDistributionPi(xi)
            #(-npl.norm(self.Y - self.A@xi)**2)/2 - self.Lambda * npl.norm(self.D@xi + self.sh,ord=1)
        return l_tab

    # --- #
    def LogDistributionPi_Full(self, x):
        return ( (-npl.norm(self.Y - self.A@x)**2)/2 , -self.Lambda*npl.norm(self.D@x + self.sh,ord=1) )


    # --- #    
    def sub_diff(self, x):
        sub = np.zeros_like(x)
        condition_pos = (x + self.sh > 0)
        condition_neg = (x + self.sh < 0)
        
        sub[condition_pos] = 1
        sub[condition_neg] = -1
        return sub
        

    # --- #
    def CalculSubdiff(self, theta, gamma):
        tmp = theta 
        tmp -= (1/2) * gamma * self.C @ (self.A.T) @ (self.Y-self.A@theta) 
        tmp -= (gamma/2) * self.C @ (self.D.T) @ self.sub_diff(self.D@theta)
        return tmp

    # --- #
    def ReturnTheta(self, theta, gamma):
        return theta

    # --- #
    def LogAlpha_NotSubdiff(self, candidate, theta, mu, gamma):
        log_alpha = self.LogDistributionPi(candidate) - self.LogDistributionPi(theta)
        return log_alpha

    # --- #
    def LogAlpha_IsSubdiff(self, candidate, theta, mu, gamma):
        log_alpha = self.LogDistributionPi(candidate) - self.LogDistributionPi(theta) 
        log_alpha -= np.log(sps.multivariate_normal.pdf(candidate, mu, gamma*self.C))
        log_alpha += np.log(sps.multivariate_normal.pdf(theta, self.CalculSubdiff(candidate, gamma), gamma*self.C))
        return log_alpha

        
    # ---------------- # ---------------- #
    # MetropolisHastings - Initialization #
    # ---------------- # ---------------- # 
    def HMInit(self, Lambda, Y, is_source, is_image, is_subdiff):
        self.Lambda = Lambda
        self.Y = Y

        # Covariance matrix C
        if is_image:
            D_1 = npl.solve(self.D, np.identity(self.T))
            self.C = D_1@D_1.T
        elif is_source:
            self.C = np.identity(self.T)
        else:
            raise Exception("method must be either 'source' or 'image' (subdiff or not)")

        # Mean vector mu
        if is_subdiff:
            self.MeanProposal = self.CalculSubdiff  #(self, theta, gamma)
        else:
            self.MeanProposal = self.ReturnTheta

        # Proposal ratio log_alpha
        if not(is_subdiff):
            self.LogRatio = self.LogAlpha_NotSubdiff  #(self, candidate, theta, mu, gamma)
        else:
            self.LogRatio = self.LogAlpha_IsSubdiff


    # --------------- # --------------- #
    # MetropolisHastings - Full version #
    # --------------- # --------------- #  
    def MetropolisHastingsFull(self, Lambda, Y, niter=1e5, method="source", save=True, SetInit=None:

        # Check the method
        is_source = method in ["source", "subdiff_source"]
        is_image = method in ["image", "subdiff_image"]
        is_subdiff = "subdiff" in method

        # Dictionary for initial values
        if SetInit is None:
            SetInit = dict()
        gamma = SetInit.get('gamma_init', 0.001)
        accept_final = SetInit.get('accept_final', 0.24)
        theta = SetInit.get('theta_init', 10 * np.ones(self.T)) # maybe choose another starting point

        # Initialization
        self.HMInit(Lambda, Y, is_source, is_image, is_subdiff)

        theta_tab = np.empty((int(niter+1), self.T))
        theta_tab[0,:] = theta
        theta_tilde_tab = np.empty((int(niter+1), self.T))
        theta_tilde_tab[0,:] = self.D @ theta

        # Burn-in aux variables
        burn_in = True
        wait_conv = False
        acceptance_cnt = 0
        rd = npr.uniform(0, 1, int(niter+1))
        
        theta_mean = np.zeros(self.T)
        cnt = 0
        converge=0
        
        L1_tab = np.empty(int(niter+1))
        L2_tab = np.empty(int(niter+1))
        L1_tab[0], L2_tab[0] = self.LogDistributionPi_Full(theta_tab[0,:])
 
        # for plotting
        if save:
            cpt = 0
            gammas = np.empty(int(niter/2))
            gammas[cpt] = gamma
            accepts = np.empty(int(niter/2))
            accepts[cpt] = 0
        else:
            gammas = accepts = None

        ## burn-in loop
        for i in range(int(niter/2)):
            mu = self.MeanProposal(theta, gamma)
            candidate = npr.multivariate_normal(mu, gamma*self.C)
            log_alpha = self.LogRatio(candidate, theta, mu, gamma)
            if log_alpha >=0:
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
                        end_burn_in = i
                        if save:
                            gammas[cpt] = gamma
                            accepts[cpt] = accept_rate
                        break
                if save:
                    gammas[cpt] = gamma
                    accepts[cpt] = accept_rate
                    cpt += 1
                acceptance_cnt = 0
            
            theta_mean += theta
            cnt += 1
            theta_tab[i+1,:] = theta
            L1_tab[i+1], L2_tab[i+1] = self.LogDistributionPi_Full(theta)
            theta_tilde_tab[i+1,:] = self.D @ theta

        if end_burn_in is None:
            raise ValueError("More iterations required")
        print("End of the burn-in")

        ## convergence loop
        for i in range(end_burn_in,int(niter)):
            mu = self.MeanProposal(theta, gamma)
            candidate = npr.multivariate_normal(mu, gamma*self.C)
            log_alpha = self.LogRatio(candidate, theta, mu, gamma)
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
            L1_tab[i+1], L2_tab[i+1] = self.LogDistributionPi_Full(theta)
            theta_tilde_tab[i+1,:] = self.D @ theta

        theta_mean /= cnt  
        return theta_tab, theta_tilde_tab, accepts, gammas, theta_mean, L1_tab, L2_tab, end_burn_in


    # ----------------- # ---------------- #
    # MetropolisHastings - Classic version #
    # ----------------- # ---------------- #  
    def MetropolisHastings(self, Lambda, Y, niter=1e5, method="source", save=True, SetInit=None):
        # MH + theta_tab

        # Check the method
        is_source = method in ["source", "subdiff_source"]
        is_image = method in ["image", "subdiff_image"]
        is_subdiff = "subdiff" in method

        # Dictionary for initial values
        if SetInit is None:
            SetInit = dict()
        gamma = SetInit.get('gamma_init', 0.001)
        accept_final = SetInit.get('accept_final', 0.24)
        theta = SetInit.get('theta_init', 10 * np.ones(self.T)) # maybe choose another starting point

        # Initialization
        self.HMInit(Lambda, Y, is_source, is_image, is_subdiff)

        theta_tab = np.empty((int(niter+1), self.T))
        theta_tab[0,:] = theta
        theta_tilde_tab = np.empty((int(niter+1), self.T))
        theta_tilde_tab[0,:] = self.D @ theta

        # Burn-in aux variables
        burn_in = True
        wait_conv = False
        acceptance_cnt = 0
        rd = npr.uniform(0, 1, int(niter+1))

        theta_mean = np.zeros(self.T)
        cnt = 0
        converge=0
 
        # for plotting
        if save:
            cpt = 0
            gammas = np.empty(int(niter/2))
            gammas[cpt] = gamma
            accepts = np.empty(int(niter/2))
            accepts[cpt] = 0
        else:
            gammas = accepts = None
    
        ## burn-in loop
        for i in range(int(niter/2)):
            mu = self.MeanProposal(theta, gamma)
            candidate = npr.multivariate_normal(mu, gamma*self.C)
            log_alpha = self.LogRatio(candidate, theta, mu, gamma)
            if log_alpha >=0 :
                theta = candidate
                acceptance_cnt += 1
            else:
                if rd[i] <= np.exp(log_alpha): # probability alpha of success
                    theta = candidate
                    acceptance_cnt += 1
                        
            # burn-in
            if ((i+1) % 1000) == 0:
                if burn_in:
                    gamma += (acceptance_cnt / 1000 - accept_final) * gamma
                    burn_in = abs(acceptance_cnt / 1000 - accept_final) > 1e-2
                    wait_conv = not burn_in
                elif wait_conv:
                    converge += 1
                    wait_conv = converge < 2e-4 * niter
                    gamma += (acceptance_cnt / 1000 - accept_final) * gamma
                    if not(wait_conv):
                        if save:
                            gammas[cpt] = gamma
                            accepts[cpt] = acceptance_cnt/1000
                        end_burn_in = i
                        break
                if save:
                    gammas[cpt] = gamma
                    accepts[cpt] = acceptance_cnt/1000
                    cpt += 1
                acceptance_cnt = 0
                
            theta_mean += theta
            cnt += 1
            theta_tab[i+1,:] = theta
            theta_tilde_tab[i+1,:] = self.D @ theta

        if end_burn_in is None:
            raise ValueError("More iterations required")
        print("End of the burn-in")

        ## convergence loop
        for i in range(end_burn_in,int(niter)):
            mu = self.MeanProposal(theta, gamma)
            candidate = npr.multivariate_normal(mu, gamma*self.C)
            log_alpha = self.LogRatio(candidate, theta, mu, gamma)
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
            theta_tilde_tab[i+1,:] = self.D @ theta

        theta_mean /= cnt  
        return theta_tab, theta_tilde_tab, accepts, gammas, theta_mean, end_burn_in


    # ----------------- # ---------------- #
    # MetropolisHastings - Fast version #
    # ----------------- # ---------------- #  
    def MetropolisHastingsFast(self, Lambda, Y, niter=1e5, method="source", SetInit=None):
        """
        estimates theta and theta_tilde using the Metropolis Hastings algorithm (MH), with burn-in
        parameters:
        - Lambda: parameter for the pi distribution
        - Y: parsimonious random vector of size T
        - SetInit : dico for initial values : gamma_init, theta_init, accept_final
        - covariance
            "source": the MH will use the identity matrix and simulate theta in the source domain
            "image": the MH will simulate theta_tilde in the image domain
        returns a tuple of two T sized vectors; means of theta and theta_tilde, obtained on all MH iterations
        """

        # Check the method
        is_source = method in ["source", "subdiff_source"]
        is_image = method in ["image", "subdiff_image"]
        is_subdiff = "subdiff" in method

        # Dictionary for initial values
        if SetInit is None:
            SetInit = dict()
        gamma = SetInit.get('gamma_init', 0.001)
        accept_final = SetInit.get('accept_final', 0.24)
        theta = SetInit.get('theta_init', 10 * np.ones(self.T)) # maybe choose another starting point

        # Initialization
        self.HMInit(Lambda, Y, is_source, is_image, is_subdiff)

        # Burn-in aux variables
        burn_in = True
        wait_conv = False
        acceptance_cnt = 0
        rd = npr.uniform(0, 1, int(niter+1))

        theta_mean = np.zeros(self.T)
        cnt = 0
        converge=0
    
        ## burn-in loop
        for i in range(int(niter/2)):
            mu = self.MeanProposal(theta, gamma)
            candidate = npr.multivariate_normal(mu, gamma*self.C)
            log_alpha = self.LogRatio(candidate, theta, mu, gamma)
            if log_alpha >=0 :
                theta = candidate
                acceptance_cnt += 1
            else:
                if rd[i] <= np.exp(log_alpha): # probability alpha of success
                    theta = candidate
                    acceptance_cnt += 1
                        
            # burn-in
            if ((i+1) % 1000) == 0:
                if burn_in:
                    gamma += (acceptance_cnt / 1000 - accept_final) * gamma
                    burn_in = abs(acceptance_cnt / 1000 - accept_final) > 1e-2
                    wait_conv = not burn_in
                elif wait_conv:
                    converge += 1
                    wait_conv = converge < 2e-4 * niter
                    gamma += (acceptance_cnt / 1000 - accept_final) * gamma
                    if not(wait_conv):
                        end_burn_in = i
                        break
                acceptance_cnt = 0
            
            theta_mean += theta
            cnt += 1
            theta_tab[i+1,:] = theta
            theta_tilde_tab[i+1,:] = self.D @ theta

        if end_burn_in is None:
            raise ValueError("More iterations required")
        print("End of the burn-in")

        ## convergence loop
        for i in range(end_burn_in,int(niter)):
            mu = self.MeanProposal(theta, gamma)
            candidate = npr.multivariate_normal(mu, gamma*self.C)
            log_alpha = self.LogRatio(candidate, theta, mu, gamma)
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
        return theta, theta_tilde