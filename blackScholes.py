import numpy as np
from scipy.stats import norm 

def BlackScholes(St : float, K : float, T : float , Sigma : float, R : float , type : str ) :
    d1 = (1/(Sigma*np.sqrt(T)))*(np.log(St/K)+(R+0.5*(Sigma)**2)*T)
    d2= d1 - Sigma*np.sqrt(T)
    if type == "Call" :
        return St*norm.cdf(d1) - K*np.exp(-R*T)*norm.cdf(d2)
    elif type == "Put" : 
        return -St*norm.cdf(-d1) + K*np.exp(-R*T)*norm.cdf(-d2)