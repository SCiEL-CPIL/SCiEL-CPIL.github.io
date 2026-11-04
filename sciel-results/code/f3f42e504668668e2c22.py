import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Define the chirp sequence generator
    def generate_h(params):
        phi0, mu = params
        k = np.arange(N)
        # h_k = cos(phi0 + mu * k^2)
        # We might also want to allow a linear frequency component, but the directive 
        # specifies "linearly varying frequency" which usually implies quadratic phase (chirp).
        # h_k = cos(phi0 + mu * k^2) is a standard chirp.
        h = np.cos(phi0 + mu * k**2)
        return h

    def objective(params):
        h = generate_h(params)
        sum_h = np.sum(h)
        # Avoid division by zero or near-zero
        if abs(sum_h) < 1e-6:
            return 1e10
        
        # Compute autoconvolution
        # np.convolve(h, h) computes full convolution
        conv_h = np.convolve(h, h)
        max_conv = np.max(conv_h)
        
        value = abs(2 * N * max_conv / (sum_h ** 2))
        return value

    # Bounds for parameters
    # phi0: 0 to 2*pi
    # mu: The chirp rate. k goes from 0 to 399. 
    # If mu is too large, the sequence oscillates rapidly and sum_h -> 0.
    # If mu is small, it's close to constant.
    # Let's search mu in a range that allows some oscillation.
    # Phase change over the array: mu * (N-1)^2.
    # For "chirp-like" behavior, we want the phase to change significantly.
    # Let's try mu in [0, 0.1] or similar.
    # Actually, let's look at the frequency. f_k = mu * 2k / (2*pi)? No, phase is mu k^2.
    # Instantaneous frequency is proportional to mu * k.
    # Let's set bounds for mu to be small enough to keep sum_h reasonable.
    # If mu = 0, h is constant cos(phi0). 
    # If cos(phi0) = 1, sum_h = N, value = 2.
    # If cos(phi0) = -1, sum_h = -N, value = 2.
    # If we introduce oscillation, max_conv might drop, but sum_h might drop faster.
    
    # Let's try a wider search.
    bounds = [(0, 2 * np.pi), (0, 0.05)]
    
    # Differential Evolution
    result_de = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-4)
    
    # Nelder-Mead refinement
    result_nm = minimize(objective, result_de.x, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = result_nm.x
    h = generate_h(best_params)
    
    # Safety check: if the result is worse than seed, return seed
    sum_h = np.sum(h)
    if abs(sum_h) < 1e-6:
        return np.ones(N)
        
    conv_h = np.convolve(h, h)
    max_conv = np.max(conv_h)
    val = abs(2 * N * max_conv / (sum_h ** 2))
    
    if val < 2.0:
        return h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
