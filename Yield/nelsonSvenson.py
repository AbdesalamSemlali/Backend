import numpy as np
from scipy.optimize import minimize


def nelson_siegel_svenson(initial_params, maturities, observed_yields,newMaturities=None) :
    if newMaturities is None :
        newMaturities = maturities 
    result = minimize(nelson_siegel_error, initial_params, args=(maturities, observed_yields))
    estimated_params = result.x
    estimated_yields = nelson_siegel_yield_curve(estimated_params, newMaturities)
    return estimated_yields

def nelson_siegel_yield_curve(params, maturities):
    
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-maturities / tau1)) / (maturities / tau1)
    term3 = beta2 * ((1 - np.exp(-maturities / tau1)) / (maturities / tau1) - np.exp(-maturities / tau1))
    term4 = beta3 * ((1 - np.exp(-maturities / tau2)) / (maturities / tau2) - np.exp(-maturities / tau2))
    
    return term1 + term2 + term3 + term4

def nelson_siegel_error(params, maturities, observed_yields):
    model_yields = nelson_siegel_yield_curve(params, maturities)
    errors = model_yields - observed_yields
    return np.sum(errors**2)
