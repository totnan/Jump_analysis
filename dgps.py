#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def dgp_Maneesoonthorn(mu = 0,kappa = 5, rho = -0.5, theta = 0.16, sigma_v = 0.5, sec = 300):
    """
    This data generating process is based on Maneesoonthorn et al. (2020)
    """
    
    dt = 1/21600

    #Generate multivariate normal variables
    W = np.random.multivariate_normal(mean = [0, 0],
                                      cov = [[dt,dt* rho],[dt* rho,dt]],
                                      size = 21600)

    #Define the stepsize
 
    #Generate the volatility process
    v = np.zeros(21600)
    #First value equals as per the paper
    v[0] = theta
    
    #Define the mean reversion parameter / Use one of Dimitru's value

    for i in range(1,21600):
        #The multiplier of sqrt(dt) comes from the discretization of a Brownian motion
        v[i] =  v[i-1] + kappa * (theta - v[i - 1] ) * dt + sigma_v * np.sqrt(v[i-1]) * W[i,0] 

    #Generate the returns
    
    r = np.zeros(21600)
    for i in range(21600):
        r[i] = mu * dt + np.sqrt(v[i]) *  W[i,1]

    #Take the cummulative sum of the return process and take every "sec" th element
    r = np.add.reduceat(r, np.arange(0,len(r),sec))
    
    return r

def dgp_SV1F(alpha = -0.100, corr = -0.62):
    """
    This DGP is based on the SV1F proposed in Dumitru and Urga (2012). The correlation parameter is fixed 
    at -0.62 in the original paper,however for parameter alpha 3 different values were used. This way we 
    only simulate the continuous part, that is used for example in the test size analysis.

    Output - np.array of length 72
    """

    #Generate multivariate normal variables
    W = np.random.multivariate_normal(mean = [0, 0],
                                      cov = [[1,-0.62],[-0.62,1]],
                                      size = 21600)

    #Define the stepsize
    dt = 1/21600

    #Generate the volatility process
    v = np.zeros(21600)
    v[0] = (0.16)
    #Define the mean reversion parameter / Use one of Dimitru's value
    alpha = -0.100
    for i in range(1,21600):
        #The multiplier of sqrt(dt) comes from the discretization of a Brownian motion
        v[i] = v[i - 1] + alpha * v[i - 1]*dt + np.sqrt(dt) * W[i,0]

    #Generate the price process
    p = np.zeros(21600)
    for i in range(21600):
        p[i] = 0.03 * dt + np.exp(0.125 * v[i]) * np.sqrt(dt) * W[i,1]

    #Take the cummulative sum of the price process and take every 300th element
    p = np.add.reduceat(p, np.arange(0,len(p),300))
    return p
