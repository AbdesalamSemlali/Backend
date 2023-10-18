import numpy as np

def american_fast_tree(St : float, K : float, T : float , Sigma : float, R : float , type : str, N : int):
    #precompute values
    dt = T/N
    u= np.exp(Sigma*np.sqrt(dt))
    d=1/u
    q = (np.exp(R*dt) - d)/(u-d)
    disc = np.exp(-R*dt)
    
    # initialise stock prices at maturity
    S = St * d**(np.arange(N,-1,-1)) * u**(np.arange(0,N+1,1))
        
    # option payoff 
    if type == 'Put':
        C = np.maximum(0, K - S)
    else:
        C = np.maximum(0, S - K)
    
    # backward recursion through the tree
    for i in np.arange(N-1,-1,-1):
        S = St * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1))
        C[:i+1] = disc * ( q*C[1:i+2] + (1-q)*C[0:i+1] )
        C = C[:-1]
        if type == 'Put':
            C = np.maximum(C, K - S)
        else:
            C = np.maximum(C, S - K)
                
    return C[0]