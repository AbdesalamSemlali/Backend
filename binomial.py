import numpy as np 

def BinomialModel(St : float, K : float, T : float , Sigma : float, R : float , type : str, N : int):
    # precompute constants
    dt = T / N
    u= np.exp(Sigma*np.sqrt(dt))
    d=1/u
    q = (np.exp(R * dt) - d) / (u - d)
    disc = np.exp(-R * dt)
    
    # initialise asset prices at maturity - Time step N
    C = St * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N+1, 1)) 
    
    # initialise option values at maturity
    if type == "Call" :
        C = np.maximum(C - K, np.zeros(N+1))
    else :
        C = np.maximum(K - C, np.zeros(N+1))
        
    # step backwards through tree
    for i in np.arange(N, 0, -1):
        C = disc * (q * C[1:i+1] + (1 - q) * C[0:i])
    
    return C[0]