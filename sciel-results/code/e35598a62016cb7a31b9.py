import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score(params):
    """
    params: [b, c1, c2, s1, s2]
    h = (1.0 * w1 + b * w2)^2
    w1, w2 are Gaussian pulses.
    We fix a=1.0 due to scale invariance.
    Returns the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    b, c1, c2, s1, s2 = params
    
    # Ensure widths are positive and reasonable
    s1 = max(abs(s1), 0.002)
    s2 = max(abs(s2), 0.002)
    
    # Clip centers to domain to keep pulses relevant
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    # Compute Gaussian windows
    # We use a sharper decay to encourage disjointness if centers are close
    # exp(-((x-c)/s)^2) is standard, let's stick to that but ensure separation in bounds
    
    w1 = np.exp(-((x - c1) / s1) ** 2)
    w2 = np.exp(-((x - c2) / s2) ** 2)
    
    # Sum of weighted windows
    # a is fixed to 1.0
    f = 1.0 * w1 + b * w2
    
    # Squared modulus
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * dx
    if l1 <= 0:
        return -1.0
        
    l2_sq = np.sum(h**2) * dx
    
    # Autoconvolution
    # Using np.convolve for small arrays is fast enough
    conv = np.convolve(h, h, mode='full') * dx
    sup_norm = np.max(conv)
    
    if sup_norm <= 0:
        return -1.0
        
    return l2_sq / (l1 * sup_norm)

def _neg_score(params):
    s = _compute_score(params)
    if s < 0:
        return 1e10
    return -s

def construct():
    # Parameters: b, c1, c2, s1, s2
    # b: amplitude ratio (a is fixed to 1)
    # c1, c2: centers
    # s1, s2: widths
    
    # To maximize the ratio, we generally want disjoint supports.
    # The domain is [-0.25, 0.25].
    # Let's encourage c1 in [-0.25, 0] and c2 in [0, 0.25] or vice versa.
    # Since the problem is symmetric, we can fix c1 < c2.
    
    bounds = [
        (0.0, 10.0),   # b (amplitude ratio)
        (-0.25, 0.0),  # c1 (left center)
        (0.0, 0.25),   # c2 (right center)
        (0.005, 0.15), # s1 (width)
        (0.005, 0.15)  # s2 (width)
    ]
    
    try:
        # Use differential evolution for global search
        # popsize=20, maxiter=100 should be fast enough
        result = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-6, 
            seed=42,
            init='sobol'
        )
        best_params = result.x
        best_score = -result.fun
    except Exception as e:
        # Fallback: Random search
        np.random.seed(42)
        best_score = -np.inf
        best_params = None
        
        for _ in range(5000):
            b = np.random.uniform(0.1, 5.0)
            c1 = np.random.uniform(-0.2, -0.05)
            c2 = np.random.uniform(0.05, 0.2)
            s1 = np.random.uniform(0.01, 0.1)
            s2 = np.random.uniform(0.01, 0.1)
            params = [b, c1, c2, s1, s2]
            s = _compute_score(params)
            if s > best_score:
                best_score = s
                best_params = params
        
        if best_params is None:
            # Default reasonable guess
            best_params = [1.0, -0.125, 0.125, 0.05, 0.05]

    b, c1, c2, s1, s2 = best_params
    s1 = max(abs(s1), 0.002)
    s2 = max(abs(s2), 0.002)
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    w1 = np.exp(-((x - c1) / s1) ** 2)
    w2 = np.exp(-((x - c2) / s2) ** 2)
    
    f = 1.0 * w1 + b * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
