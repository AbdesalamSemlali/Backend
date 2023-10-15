import math
import numpy as np

def TrinomialModel(St : float, K : float, T : float , Sigma : float, R : float , type : str, N : int):
    """
    Trinomial option pricing model for European options.

    Parameters:
    St (float): Current stock price
    K (float): Option strike price
    T (float): Time to maturity (in years)
    R (float): Risk-free interest rate
    sigma (float): Volatility of the underlying stock
    N (int): Number of time steps
    type (str): 'call' for call option, 'put' for put option

    Returns:
    float: Option price
    """

    dt = T / N
    nu = R - 0.5 * Sigma**2
    nu_dt = nu * dt
    sqrt_dt = math.sqrt(dt)
    
    # Calculate up, down, and no-move factors
    u = math.exp(Sigma * sqrt_dt)
    d = 1 / u
    m = 1
    
    # Calculate probabilities
    pu = ((math.exp(nu_dt) - d) / (u - d))**2
    pd = ((u - math.exp(nu_dt)) / (u - d))**2
    pm = 1 - pu - pd
    
    # Initialize option value at maturity
    option_values = np.zeros((2 * N + 1, 2 * N + 1))

    # Initialize option value at maturity
    j = np.arange(-N, N + 1)
    option_values[N] = np.maximum(0, St * u**j * d**(2 * np.abs(j) - N) - K)

    # Backward iteration to calculate option price at each step
    for i in range(N - 1, -1, -1):
        j = np.arange(-i, i + 1)
        option_values[i, j] = np.exp(-R * dt) * (pu * option_values[i + 1, j + 1] + pm * option_values[i + 1, j] + pd * option_values[i + 1, j - 1])

    return option_values[0, 0] if type == 'call' else np.maximum(0, K - St)

