import numpy as np
from scipy.stats import norm




def delta(St : float, K : float, T : float , Sigma : float, R : float , type : str) : 
    d1 = (1/(Sigma*np.sqrt(T)))*(np.log(St/K)+(R+0.5*(Sigma)**2)*T)
    return norm.cdf(d1) if type=="Call" else -norm.cdf(-d1)

def gamma(St : float, K : float, T : float , Sigma : float, R : float , type : str) :
    d1 = (1/(Sigma*np.sqrt(T)))*(np.log(St/K)+(R+0.5*(Sigma)**2)*T)
    return norm.pdf(d1)/(St*Sigma*np.sqrt(T)) 

def theta(St : float, K : float, T : float , Sigma : float, R : float , type : str) :
    d1 = (1/(Sigma*np.sqrt(T)))*(np.log(St/K)+(R+0.5*(Sigma)**2)*T)
    d2 = d1-Sigma*np.sqrt(T)
    call = -St*norm.pdf(d1, 0, 1)*Sigma/(2*np.sqrt(T)) - R*K*np.exp(-R*T)*norm.cdf(d2, 0, 1)
    put = (-St*Sigma*norm.pdf(d1))/(2*np.sqrt(T)) + R*K*np.exp(-R*T)*norm.cdf(-d2)
    return  call/365 if type=="Call" else put/365

def vega(St : float, K : float, T : float , Sigma : float, R : float , type : str) :
    d1 = (1/(Sigma*np.sqrt(T)))*(np.log(St/K)+(R+0.5*(Sigma)**2)*T)
    d2 = d1-Sigma*np.sqrt(T)
    return St*norm.pdf(d1)*np.sqrt(T)*0.01

def rho(St : float, K : float, T : float , Sigma : float, R : float , type : str) :
    d1 = (1/(Sigma*np.sqrt(T)))*(np.log(St/K)+(R+0.5*(Sigma)**2)*T)
    d2 = d1-Sigma*np.sqrt(T)
    call = K*T*np.exp(-R*T)*norm.cdf(d2, 0, 1)
    put = -K*T*np.exp(-R*T)*norm.cdf(-d2, 0, 1)
    return call*0.01 if type=="Call" else put*0.01