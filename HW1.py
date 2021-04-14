import numpy as np
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statistics

################################
##Simulate DGP data 1000 times##
################################

def DGP(N=1000, M=1000, alpha=0, beta_1=1, beta_2=0.1, rho=0.9):
    seed = 1234
    random.seed(1234)

    coefX_1 = []
    conf_int = []
    num_x = []
    X_1 = []
    
    for i in range(M):
        #generate data first
        X = np.random.multivariate_normal((0,0), [[1, rho], [rho, 1]], N)
        epsilon = np.random.normal(0,1,size=(N,1))
        beta = np.array([beta_1,beta_2]).reshape(-1,1)
        Y = alpha + X@beta + epsilon
        
        #now estimate coefficient on X1
        unrmod = sm.OLS(Y,X)
        unr_b1 = unrmod.fit().summary2().tables[1]['Coef.'][0]
        
        #Choose coefficients and confidence intervals properly
        if unrmod.fit().summary2().tables[1]['P>|t|'][1] > 0.5:
            rmod = sm.OLS(Y, [l[0] for l in X])
            chosen_b1 = rmod.fit().summary2().tables[1]['Coef.'][0]
            conf_lower = rmod.fit().summary2().tables[1]['[0.025'][0]
            conf_higher = rmod.fit().summary2().tables[1]['0.975]'][0]
            chosen_conf = (conf_lower, conf_higher)
            num = N
        else:
            chosen_b1 = unr_b1
            conf_lower = unrmod.fit().summary2().tables[1]['[0.025'][0]
            conf_higher = unrmod.fit().summary2().tables[1]['0.975]'][0]
            chosen_conf = (conf_lower, conf_higher)
            num = N
            
        #Save chosen and unrestricted coefs on x1
        coefX_1.append((chosen_b1, unr_b1))
        
        #Save confidence intervals
        conf_int.append(chosen_conf)
        
        #Save X1
        X_1.append([l[0] for l in X])
        
        #Save number of X
        num_x.append(num)
        
    
    #print("The coefficients are:", coefX_1)
    #print("The confidence intervals are:", conf_int)
    #print("The number of X are:", num_x)
    #print("The X1 values are:", X_1)
    
    #Make plot of betas on X1 values
    plt.hist([l[0] for l in coefX_1], bins=25) #chosen model
    plt.hist([l[1] for l in coefX_1], alpha = 0.5, bins=25) #unrestricted model
    
    #Get percentage of cases for which confidence interval captures beta
    count = 0
    for k in conf_int:
        if k[0] <= 1 <= k[1]:
            count += 1
    percent = count / M
    print("The percentage is:", percent)
    
    
    unr_var = statistics.variance(l[1] for l in coefX_1)
    chosen_var = statistics.variance(l[0] for l in coefX_1)
    print("The variance of chosen betas is", chosen_var)
    print("The variance of unrestricted betas is", unr_var)
    
DGP(N=50)

DGP(N=100)

DGP(N=200)

DGP(N=1000)

DGP(rho=0.1)

DGP(rho=0.5)

DGP(rho=0.9)

DGP(beta_2=0.01)

DGP(beta_2=0.05)

DGP(beta_2=0.1)

DGP(beta_2=0.2)

DGP(beta_2=1)
