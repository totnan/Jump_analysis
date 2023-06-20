#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def realized_var(rt:np.array):
    "Calculate the realized variance of the returns"
    RV = np.sum(np.power(rt, 2))
    return RV

def bipower_var(rt : np.array):
    M = len(rt)
    BV = (np.pi/2) * (M/(M-1)) * np.sum(np.abs(rt[1:] *np.abs(rt[:-1])))
    return BV

def BNS(rt : np.array):
    """
    Return the tests statistic defined in Barndorff-Nielsen and Shephard (2006)
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.

    """
    M = len(rt)
    
    RV = realized_var(rt)
    BV = bipower_var(rt)
    
    constant = ((2**(2/3))*math.gamma(7/6)*(math.gamma(1/2)**(-1/2)))**(-3)
    temp = np.sum(np.abs(rt[:-2])**(4/3) * np.abs(rt[1:-1])**(4/3) * np.abs(rt[2:])**(4/3))
    
    TP = constant * (M**2/(M-2)) * temp
    
    numerator = 1 - (BV/RV)
    denumerator = np.sqrt(((np.pi/2)**2 + np.pi - 5) * M**-1 * max(1, (TP/BV**2)))
    
    test = numerator / denumerator
    
    return test

def CPR(rt: np.array,K: int, c: int):
    
    """
    Return the tests statistic defined in Corsi, Pirino, and Reno (2010) 
    
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.
                  The authors used K = 10 and c = 3
    """
    
    M = len(rt)
    RV = realized_var(rt)
    
    temp = np.multiply(np.abs(rt[:-1]), np.abs(rt[1:]))
    Vti = np.zeros(M-K-1)
    
    for i in range(M-K-1):
        Vti[i] = (np.pi/2) *(1/(K-1)) * np.sum(temp[i: i+K+1])
    
    psi = (c**2) * Vti
    r = rt[K+1:]
    
    tau1 = np.where(r <= np.sqrt(psi), np.abs(r), 1.094*np.sqrt(psi))
    tau43 = np.where(r <= np.sqrt(psi), np.abs(r)** (4/3), 1.129 * psi**(2/3))
    
    Mn = len(tau1) 
    CTBV = (np.pi/2) * (Mn / (Mn-1)) * np.sum(tau1[:-1] * tau1[1:])
    constant = ((2**(2/3))*math.gamma(7/6)*(math.gamma(1/2)**(-1/2)))**(-3)
    
    temp2 = np.sum(tau43[0:-2]*tau43[1:-1]*tau43[2:])
    CTPV = constant * (Mn**2 / (Mn-2)) * temp2
    
    numerator = 1 - (CTBV/RV)
    denumerator = ((np.pi/2) **2 + np.pi - 5) * (1/Mn) * max(1, CTPV/CTBV**2)
    
    test = numerator / np.sqrt(denumerator)
    
    return test

def TMIN(rt : np.array):  
    """
    Andersen, Dobrev, and Schaumburg (2009)
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.
    """
    
    #Length of the (log) return series
    M = len(rt)
    
    #Calculate realized variance
    RV = realized_var(rt)
    
    temp = np.minimum(np.abs(rt[1:]),np.abs(rt[:-1]))
    MinRV = (np.pi/(np.pi-2))*(M/(M-1)) *np.sum(temp**2)
    
    MinRQ = (np.pi/(3*np.pi-8))* (M**2/(M-1)) *np.sum(temp**4)
    
    numerator = 1 - (MinRV/RV)
    denumerator = np.sqrt(1.81/M) * max(1,(MinRQ/(MinRV**2)))

    test = numerator / denumerator
    return test

def TMED(rt : np.array):
    """
    Andersen, Dobrev, and Schaumburg (2009)
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.
    """
    M = len(rt)
    RV = realized_var(rt)
    data_med = np.stack([np.abs(rt[0:M-2]), np.abs(rt[1:M-1]), np.abs(rt[2:])])
    constant_medrv = np.pi /(np.pi + 6 - 4*np.sqrt(3)) * (M/(M-2))
    MedRV = constant_medrv * np.sum(np.median(data_med, axis = 0)**2)
    
    constant_medrq = 3*np.pi/(9*np.pi + 72 - 52*np.sqrt(3)) * (M**2/(M-2))
    MedRQ = constant_medrq * np.sum(np.median(data_med,axis = 0)**4)
    
    numerator = 1 - (MedRV/RV)
    denumerator = np.sqrt(0.96/M *max(1,(MedRQ/MedRV**2)))
    test = numerator / denumerator
    
    return test

def PZ2(rt : np.array, tau: int):
    """
    Podolskij and Ziggel (2010) 
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.
    
    """
    M = len(rt)
    BV = bipower_var(rt)
    trunc = 2.3 * np.sqrt(BV) * ((1/M)**(0.4))

    unif = np.random.uniform(size = M)
    eta = np.where(unif <= 0.5, 1-tau, 1+tau)

    Bhat2 = np.sqrt(M) * np.sum(np.abs(rt)**2 * (1 - eta *np.where(np.abs(rt) < trunc, 1, 0)))
    Vbar4 = M * np.sum(np.where(np.abs(rt) < trunc,rt **4,0))
    var2 = np.var(eta, ddof = 1) * Vbar4 #Need to set degrees of freedom to 1
    
    test = Bhat2/np.sqrt(var2)
    
    return test

def PZ4(rt : np.array, tau: int):
    """
    Podolskij and Ziggel (2010) 
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.
    Tau?
    """
    M = len(rt)
    BV = bipower_var(rt)
    trunc = 2.3 * np.sqrt(BV) * ((1/M)**(0.4))

    unif = np.random.uniform(size = M)
    eta = np.where(unif <= 0.5, 1-tau, 1+tau)

    Bhat4 =M**(3/2) * np.sum(np.abs(rt)**4 * (1 - eta *np.where(np.abs(rt) < trunc, 1, 0)))
    Vbar8 = M**3 * np.sum(np.where(np.abs(rt) < trunc,rt **8,0))
    var4 = np.var(eta,ddof = 1) * Vbar8
    
    test = Bhat4/np.sqrt(var4)
    
    return test

def ASJ(rt : np.array):
    """
    Ait-Sahalia and Jacod (2008) 
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.
    """
    M = len(rt)
    K= 3
    temp = np.multiply(np.abs(rt[:-1]), np.abs(rt[1:]))
    Vti = np.zeros(M-K-1)
    
    for i in range(M-K-1):
        Vti[i] = (np.pi/2) *(1/(K-1)) * np.sum(temp[i: i+K+1])

    mp = 3
    m2p = 105
    Mpk = (16 * 2) * (2*4-2-1)/3

    tempr = rt[M-len(Vti):M]
    #Define M for further calculation
    M = len(tempr)
    rtk = tempr[1::2]
    
    Bhat = np.sum(np.abs(np.power(tempr,4)))
    Bhatk = np.sum(np.abs(np.power(rtk,4)))
    Shat = Bhat / Bhatk
    
    trunc = 3*np.sqrt(Vti*M)*((1/M)**0.48)
    Ahat = (M/mp) * np.sum(np.abs(np.power(tempr,4)) * np.where(np.abs(tempr) < trunc,1,0))
    Ahat2 = ((M**3)/m2p) * np.sum(np.abs(np.power(tempr,8)) * np.where(np.abs(tempr) < trunc,1,0))
    
    Sig = (1/M) * Mpk * Ahat2 / (Ahat**2)
    test = (Shat - 2) /np.sqrt(Sig)
    
    return test

def ABD(rt : np.array):
    """
    Andersen, Bollerslev, and Dobrev (2007)
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.
    """
    M = len(rt)
    BV = bipower_var(rt)
    
    vals = rt / np.sqrt(BV/M)
    return np.abs(np.max(vals))

def LM(rt):
    """
    Lee and Mykland (2008) 
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.
    """
    M = len(rt)
    K = 2
    
    
    temp = np.multiply(np.abs(rt[:-1]), np.abs(rt[1:]))
    Vti = np.zeros(M-K-1)
    
    for i in range(M-K-1):
        Vti[i] = (np.pi/2) *(1/(K-1)) * np.sum(temp[i: i+K+1])
    
    tempr = rt[M-len(Vti): M]
    Ttil = np.abs(tempr) / np.sqrt(Vti)
    
    Cm = (np.sqrt(2*np.log(M))/0.8)-((np.log(np.pi)+np.log(np.log(M)))/(1.6*np.sqrt(2*np.log(np.pi))))
    Sm=1/(0.6*np.sqrt(2*np.log(np.pi)))
    LM=(np.max(Ttil)-Cm)/Sm;
    
    return LM

def JO(log_returns:np.array):
    """
    Jiang and Oomen (2008) Test
    
    Code source : Maneesoonthorn, Worapree, Gael M. Martin, and Catherine S. Forbes. 
                  "High-frequency jump tests: Which test should we use?." Journal of econometrics 219.2 (2020): 478-487.
    """

    
    returns = np.exp(log_returns)-1
 
    #Length of the return series
    M = len(log_returns)

    #Initialize a temporary variable
    temp = (               
                (np.absolute(log_returns[0:len(log_returns)-3]) *
                np.absolute(log_returns[1:len(log_returns)-2]) *
                np.absolute(log_returns[2:len(log_returns)-1]) *
                np.absolute(log_returns[3:len(log_returns)])) ** (3/2)
            )

    #Calculate Sigma in the definition
    Sigma = 3.05 * ((M**3) / (M-3)) * np.sum(temp)

    # Calculate SWV in the definition
    SWV = 2* np.sum(returns - log_returns)
    
    #Calculate bipower variation and realized variance
    BV = bipower_var(log_returns)
    RV = realized_var(log_returns)

    #Calculate the test statistic
    test_stat = ((M*BV)/np.sqrt(Sigma))*(1 - (RV/SWV))
    return test_stat