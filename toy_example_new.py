import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import scipy.stats as sps
from scipy.sparse import diags

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

#simulation of Y using exponential distribution
def Computation_Y_Exp(T, Lambda, a, b, random_state=None):
    param_accept = 2/T
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T,a,b)

    if random_state is None:
        npr = np.random.default_rng()
    else:
        npr = random_state #something has been passed as param, might fail
    
    rd = npr.uniform(0, 1, T)
    x_tilde_true = np.where(rd < param_accept, npr.exponential(1/Lambda, T), 0)*(2*npr.binomial(1,1/2,T) - 1) - sh 
    x_true = npl.solve(D, x_tilde_true)
    Y = npr.multivariate_normal(A @ x_true, np.identity(T))

    return Y

#simulation of Y using deterministic x_true
def Computation_Y_Det(T, a, b, random_state=None):
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T,a,b)
    x_true = np.zeros(T)

    coeff_dir = 2.7/T
    ord_ori   = 0.1
    
    k = int(T/3)
    x_true[:k] = 2.7/T * np.arange(k) + 0.1 #vectorised for-loop - gives same result
    x_true[k:] = -3/T  * np.arange(T-k) + 2

    x_true = x_true * 10
    
    if random_state is None:
        npr = np.random.default_rng()
    else:
        npr = random_state

    Y = npr.multivariate_normal(A @ x_true, np.identity(T))

    return Y    

#compute argmax of pi and pi_tilde distributions
def ComputeArgmax(T, Lambda, Y, a, b):
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    
    u = U @ Y + sh
    x_tilde = np.sign(u)*np.maximum(np.abs(u) - Lambda, np.zeros(T)) - sh
    x = npl.solve(D,x_tilde)
    
    return x,x_tilde

#compute theoretical means
def ComputeMeans(T, Lambda, Y, a, b):
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

#compute theoretical quantiles
def ComputeQuantiles(T, Lambda, threshold, Y, a, b, niter=1e5):
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

    for j in range(T):
        q_plus = sps.norm.cdf(np.minimum(ub[:, None]-mu_plus[j], -sh[j]-mu_plus[j])) / C_plus[j]
        q_minus = np.where(ub[:, None] > -sh[j], (sps.norm.cdf(ub[:, None]-mu_minus[j]) + C_minus[j] - 1) / C_minus[j], 0)
        probas[:, j] = gamma[j] * q_plus.squeeze() + (1 - gamma[j]) * q_minus.squeeze()
    
    quantiles = ub[np.argmax(probas >= threshold, axis=0)]
        
    return quantiles

#return empirically computed quantiles
def Quantiles(sim_tab,q,T):
    return np.percentile(sim_tab,q,axis=0)

#log_pi
def LogDistributionPi(x, Y, A, D, sh, Lambda):
    return (-npl.norm(Y - A@x)**2)/2-Lambda*npl.norm(D@x + sh,ord=1)

#pi
def DistributionPi(x, Y, A, D, sh, Lambda):
    return np.exp(LogDistributionPi(x, Y, A, D, sh, Lambda))

#log_pi_tilde
def LogDistributionPi_tilde(x_tilde, Y, U, sh, Lambda):
    return (-npl.norm(U@Y - x_tilde)**2)/2-Lambda*npl.norm(x_tilde + sh,ord=1)

#pi_tilde
def DistributionPi_tilde(x_tilde, Y, U, sh, Lambda):
    return np.exp(LogDistributionPi_tilde(x_tilde, Y, U, sh, Lambda))

#log_pi for a table
def LogDistributionPi_Tab(x_tab, Y, A, D, sh, Lambda):
    return np.apply_along_axis(LogDistributionPi, 1, x_tab, Y=Y, A=A, D=D, sh=sh, Lambda=Lambda)

#log_pi_tilde for a table
def LogDistributionPi_tilde_Tab(x_tilde_tab, Y, U, sh, Lambda):
    return np.apply_along_axis(LogDistributionPi_tilde, 1, x_tilde_tab, Y=Y, U=U, sh=sh, Lambda=Lambda)

#compute subdiff term in source
def SubdiffSource(theta, Y, A, D, sh, gamma):
    return theta - gamma * A.T @ (A@theta - Y) - gamma * D.T @ np.sign(D@theta + sh)

#compute subdiff term in image
def SubdiffImage(theta_tilde, Y, U, sh, gamma):
    return theta_tilde - gamma * (theta_tilde - U @ Y) - gamma * np.sign(theta_tilde + sh)

#compute proximal term in source
def DriftSource(theta, Y, A, D, sh, Lambda, gamma):
    N = len(theta)
    nabla_f = A.T@(A@theta - Y)
    D_1 = npl.solve(D,np.identity(N))
    tau = theta - gamma*nabla_f + D_1@sh
    return -D_1@sh + np.sign(tau)*np.maximum(np.abs(tau) - Lambda*gamma*D.T@np.ones(N), np.zeros(N))

#compute proximal term in image
def DriftImage(theta_tilde, Y, U, sh, Lambda, gamma):
    N = len(theta_tilde)
    nabla_f = -(U@Y-theta_tilde)
    tau = theta_tilde - gamma*nabla_f + sh
    return -sh + np.sign(tau)*np.maximum(np.abs(tau) - Lambda*gamma*np.ones(N), np.zeros(N))

#MetropolisHastings algorithm - returns all parameters + a plot
def MetropolisHastings(T, Lambda, Y, a, b, niter=1e5, method="source", random_state=None):
    #------------- Check the method -------------# 
    source_methods = ["source", "subdiff_source", "prox_source"]
    image_methods  = ["image",  "subdiff_image",  "prox_image"]

    if method in source_methods:
        is_source = True
    elif method in image_methods:
        is_source = False
    else:
        raise Exception(f"method doesn't exist. choose among {source_methods+image_methods}")
    
    is_prox = "prox" in method
    is_subdiff = "subdiff" in method

    #------------- Check for RandomState -------------#
    if random_state is None:
        npr = np.random.default_rng()
    else:
        npr = random_state

    #------------- Initialize parameters -------------# 
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)
    x, x_tilde = ComputeArgmax(T, Lambda, Y, a, b)
    mu, mu_tilde = ComputeMeans(T, Lambda, Y, a, b)
    gamma = 0.001

    accept_final = 0.24
    accept_cnt = 0
    accept_rate = None
    rd = npr.uniform(0, 1, int(niter+1))

    end_burn_in = None

    gammas = []
    accepts = []
    theta = np.linalg.solve(D, np.random.uniform(-1, 1, T)) #initial point
    theta_tab = np.empty((int(niter) + 1, T))
    fig,ax = plt.subplots(1,1)


    #------------- Case: Source -------------#
    if is_source:
        ax.set_ylabel("Value")
        ax.set_xlabel(r"Component of $\theta$ / time axis")
        ax.set_title(r"Burn-in of $\theta$ using method: "+str(method))
        ax.plot(x, color="salmon", label="argmax")
        ax.plot(mu, color="purple", label="theoretical mean")
        theta_tab[0] = theta #we save the initial point
        for i in range(int(niter/2)):
            if is_subdiff:
                mu = SubdiffSource(theta, Y, A, D, sh, gamma)
            elif is_prox:
                mu = DriftSource(theta, Y, A, D, sh, Lambda, gamma)
            else:
                mu = theta

            candidate = npr.multivariate_normal(mu, gamma*np.identity(T))
            if not is_prox:
                log_alpha = LogDistributionPi(candidate, Y, A, D, sh, Lambda) - LogDistributionPi(theta, Y, A, D, sh, Lambda)
            else:
                mu_cand = DriftSource(candidate, Y, A, D, sh, Lambda, gamma)
                log_alpha = LogDistributionPi(candidate, Y, A, D, sh, Lambda) - LogDistributionPi(theta, Y, A, D, sh, Lambda) - npl.norm(candidate - mu)**2/(2*gamma) + npl.norm(theta - mu_cand)**2/(2*gamma)
           
            if log_alpha >= 0 :
                theta = candidate
                accept_cnt += 1
            elif rd[i] <= np.exp(log_alpha): #probability alpha of success
                theta = candidate
                accept_cnt += 1
            
            theta_tab[i+1] = theta

            #burn-in
            if ((i+1) % 1000) == 0:
                plt.plot(theta, "r", alpha = 2*i/niter)
                accept_rate = accept_cnt/1000
                gammas.append(gamma)
                accepts.append(accept_rate)
                accept_cnt = 0 #reset
                gamma += (accept_rate - accept_final) * gamma
                
                if abs(accept_rate - accept_final) <= 1e-2:
                    end_burn_in = i
                    break
                
        if end_burn_in is None:
            end_burn_in = int(niter/2)-1
        print(f"ended burn-in @ {end_burn_in:d}: {accepts[-1]-accept_final:.3f}")
            
        gammas = np.array(gammas)
        accepts = np.array(accepts)

        for i in range(end_burn_in+1,int(niter)):
            if is_subdiff:
                mu = SubdiffSource(theta, Y, A, D, sh, gamma)
            elif is_prox:
                mu = DriftSource(theta, Y, A, D, sh, Lambda, gamma)
            else:
                mu = theta

            candidate = npr.multivariate_normal(mu, gamma*np.identity(T))
            if not is_prox:
                log_alpha = LogDistributionPi(candidate, Y, A, D, sh, Lambda) - LogDistributionPi(theta, Y, A, D, sh, Lambda)
            else:
                mu_cand = DriftSource(candidate, Y, A, D, sh, Lambda, gamma)
                log_alpha = LogDistributionPi(candidate, Y, A, D, sh, Lambda) - LogDistributionPi(theta, Y, A, D, sh, Lambda)  - npl.norm(candidate - mu)**2/(2*gamma) + npl.norm(theta - mu_cand)**2/(2*gamma)
            if log_alpha >= 0 :
                theta = candidate
                accept_cnt += 1
            elif rd[i] <= np.exp(log_alpha): #probability alpha of success
                theta = candidate
                accept_cnt += 1
            
            theta_tab[i+1] = theta
    
    #------------- Case: Image -------------#
    else:
        ax.set_ylabel("Value")
        ax.set_xlabel(r"Component of $\theta$ / time axis")
        ax.set_title(r"Burn-in of $\theta$ using method: "+str(method))
        ax.plot(x, color="salmon", label="argmax")
        ax.plot(mu, color="purple", label="theoretical mean")

        D_1 = npl.solve(D, np.identity(T))
        super_D = npl.solve(D_1@D_1.T, np.identity(T))
        theta = D @ theta #we save theta_tildes in the theta variable
        theta_tab[0] = theta #we save all theta_tildes in theta_tab, including initial theta_tilde
        for i in range(int(niter/2)):
            if is_subdiff:
                mu = SubdiffImage(theta, Y, U, sh, gamma)
            elif is_prox:
                mu = DriftImage(theta, Y, U, sh, Lambda, gamma)
            else:
                mu = theta
            
            #covariance matrix changes
            candidate = npr.multivariate_normal(mu, gamma * D_1@D_1.T)
            if not is_prox:
                log_alpha = LogDistributionPi_tilde(candidate, Y, U, sh, Lambda) - LogDistributionPi_tilde(theta, Y, U, sh, Lambda)
            else:
                mu_cand = DriftImage(theta, Y, U, sh, Lambda, gamma)
                log_alpha = LogDistributionPi_tilde(candidate, Y, U, sh, Lambda) - LogDistributionPi_tilde(theta, Y, U, sh, Lambda) - (theta - mu_cand).T @ super_D @ (theta - mu_cand)/(2*gamma) + (candidate - mu).T @ super_D @ (candidate - mu)/(2*gamma)
            if log_alpha >= 0 :
                theta = candidate
                accept_cnt += 1
            elif rd[i] <= np.exp(log_alpha): #probability alpha of success
                theta = candidate
                accept_cnt += 1
            
            theta_tab[i+1] = theta

            #burn-in
            if ((i+1) % 1000) == 0:
                plt.plot(D_1@theta, "r", alpha = 2*i/niter) #for coherence we choose to plot theta
                accept_rate = accept_cnt/1000
                gammas.append(gamma)
                accepts.append(accept_rate)
                accept_cnt = 0 #reset
                gamma += (accept_rate - accept_final) * gamma
                
                if abs(accept_rate - accept_final) <= 1e-2:
                    end_burn_in = i
                    break
                
        if end_burn_in is None:
            end_burn_in = int(niter/2)-1
        print(f"ended burn-in @ {end_burn_in:d}: {accepts[-1]-accept_final:.3f}")
        
        gammas = np.array(gammas)
        accepts = np.array(accepts)

        for i in range(end_burn_in+1,int(niter)):
            if is_subdiff:
                mu = SubdiffImage(theta, Y, U, sh, gamma)
            elif is_prox:
                mu = DriftImage(theta, Y, U, sh, Lambda, gamma)
            else:
                mu = theta

            candidate = npr.multivariate_normal(mu, gamma * D_1@D_1.T) #covariance matrix changes
            if not is_prox:
                log_alpha = LogDistributionPi_tilde(candidate, Y, U, sh, Lambda) - LogDistributionPi_tilde(theta, Y, U, sh, Lambda)
            else:
                mu_cand = DriftImage(theta, Y, U, sh, Lambda, gamma)
                log_alpha = LogDistributionPi_tilde(candidate, Y, U, sh, Lambda) - LogDistributionPi_tilde(theta, Y, U, sh, Lambda) - (theta - mu_cand).T @ super_D @ (theta - mu_cand)/(2*gamma) + (candidate - mu).T @ super_D @ (candidate - mu)/(2*gamma)
            
            if log_alpha >= 0 :
                theta = candidate
                accept_cnt += 1
            elif rd[i] <= np.exp(log_alpha): #probability alpha of success
                theta = candidate
                accept_cnt += 1
            
            theta_tab[i+1] = theta
        
        #now we need to convert the theta_tildes back to theta
        theta_tab = (D_1 @ theta_tab.T).T
    ax.legend()
    return theta_tab, accepts, gammas, theta_tab[end_burn_in:].mean(axis=0), end_burn_in, fig

#Specialised One-at-a-time PGdual algorithm -> returns only table of simulations and ends of burn in
def PGdual_One_at_a_time(T, Lambda, Y, a, b, niter=1e5, method="source", random_state=None):
    #------------- Check the method -------------# 
    if method=="source":
        is_source = True
    elif method=="image":
        is_source = False
    else:
        raise Exception("method must be either \'image\' or \'source\'")

    #------------- Check for RandomState -------------#
    if random_state is None:
        npr = np.random.default_rng()
    else:
        npr = random_state

    #------------- Initialize parameters -------------# 
    D = BuildD(T)
    U, Delta, Vt = BuildUVDelta(D)
    A = BuildA(Delta, Vt)
    sh = Buildsh(T, a, b)

    gamma = 0.001*np.ones(T)
    accept_final = 0.24
    accept_rate = np.zeros(T)
    accept_cnt  = np.zeros(T)
    end_burn_in = np.zeros(T)

    theta = np.linalg.solve(D, np.random.uniform(-1, 1, T)) #initial point
    theta_tab = np.empty((int(niter) + 1, T))

    #------------- Case: Source -------------#
    if is_source:
        theta_tab[0] = theta
        for i in range(int(niter)):
            mu = DriftSource(theta, Y, A, D, sh, Lambda, gamma)
            candidate = npr.multivariate_normal(mu, np.diag(gamma))
            mu_cand   = DriftSource(candidate, Y, A, D, sh, Lambda, gamma)
            log_alpha = np.apply_along_axis(LogDistributionPi, 0, candidate, Y=Y, A=A, D=D, sh=sh, Lambda=Lambda) - np.apply_along_axis(LogDistributionPi, 0, theta, Y=Y, A=A, D=D, sh=sh, Lambda=Lambda) - np.abs(candidate - mu)/(2*gamma) + np.abs(theta - mu_cand)/(2*gamma)
            rd = npr.uniform(size=T)
            
            #we create a boolean map
            bool_map = (log_alpha >= 0) | (rd <= np.exp(log_alpha))

            #update
            theta[bool_map] = candidate[bool_map]
            theta_tab[i+1] = theta
            accept_cnt += bool_map

            #burn-in
            if ((i+1) % 1000) == 0 and (end_burn_in == 0).any():
                accept_rate = accept_cnt/1000
                accept_cnt = np.zeros(T) #reset
                gamma += (accept_rate - accept_final) * gamma
                tmp_burn_in = np.abs(accept_rate - accept_final) <= 1e-2
                update_map = end_burn_in == 0
                end_burn_in[update_map] = i * tmp_burn_in[update_map]
                
    #------------- Case: Image -------------#    
    else:
        theta_tab[0] = D @ theta
        D_1 = npl.solve(D, np.identity(T))
        super_D = npl.solve(D_1@D_1.T, np.identity(T))
        for i in range(int(niter)):
            mu = DriftImage(theta, Y, U, sh, Lambda, gamma)
            candidate = npr.multivariate_normal(mu, np.diag(gamma) * D_1@D_1.T)
            mu_cand   = DriftImage(candidate, Y, U, sh, Lambda, gamma)

            log_alpha = np.apply_along_axis(LogDistributionPi_tilde, 0, candidate, Y=Y, U=U, sh=sh, Lambda=Lambda) - np.apply_along_axis(LogDistributionPi_tilde, 0, theta, Y=Y, U=U, sh=sh, Lambda=Lambda) - (theta - mu_cand) @ super_D @ (theta - mu_cand)/2*gamma + (candidate - mu) @ super_D @ (candidate - mu)/2*gamma

            rd = npr.uniform(size=T)
            
            #we create a boolean map
            bool_map = (log_alpha >= 0) | (rd <= np.exp(log_alpha))
            assert bool_map.shape==(T,) #small test
            
            #update
            theta[bool_map] = candidate[bool_map]
            theta_tab[i+1] = theta
            accept_cnt += bool_map

            #burn-in
            if ((i+1) % 1000) == 0 and (end_burn_in == 0).any():
                accept_rate = accept_cnt/1000
                accept_cnt = np.zeros(T) #reset
                gamma += (accept_rate - accept_final) * gamma
                tmp_burn_in = np.abs(accept_rate - accept_final) <= 1e-2
                update_map = end_burn_in == 0
                end_burn_in[update_map] = i * tmp_burn_in[update_map]
        #we need to convert our theta_tildes to theta
        theta_tab = (D_1 @ theta_tab.T).T
    return theta_tab, end_burn_in


