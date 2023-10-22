import numpy as np 
from myarrange import myArrange
def TrinomialModel(St : float, K : float, T : float , Sigma : float, R : float , type : str, N : int):

    dt = T / N
    u = np.exp(Sigma*np.sqrt(2*dt))
    d=1/u
    pu=(((np.exp(R*dt/2))-np.exp(-Sigma*np.sqrt(dt/2)))/
        (np.exp(Sigma*np.sqrt(dt/2))-np.exp(-Sigma*np.sqrt(dt/2))))**2
    pd=(((np.exp(Sigma*np.sqrt(dt/2)))-np.exp(R*dt/2))/
        (np.exp(Sigma*np.sqrt(dt/2))-np.exp(-Sigma*np.sqrt(dt/2))))**2
    pm=1-pu-pd
    disc = np.exp(-R * dt)

    C = St * d ** myArrange(N) * u ** myArrange(N)[::-1] 
    
    if type == "Call" :
        C = np.maximum(C - K, np.zeros(2*N+1))
    else :
        C = np.maximum(K - C, np.zeros(2*N+1))

    for i in np.arange(2*N, 0, -2):
        C = disc * (pu * C[2:i+1] + pm * C[1:i] + pd * C[0:i-1])
    
    return C[0]