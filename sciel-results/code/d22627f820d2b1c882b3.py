import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive Lower Confidence Bound (LCB) for minimization
    # score = mu - kappa * sigma
    
    # Adaptive kappa:
    # When f_best is high (poor), use high kappa (explore).
    # When f_best is low (good), use low kappa (exploit).
    # Range of f_best considered: 0 to -40 (approx min)
    
    # Clip f_best to reasonable range for scaling
    f_best_clipped = np.clip(f_best, -40, 0)
    
    # Linear interpolation
    # f_best = 0 -> kappa = 2.5
    # f_best = -40 -> kappa = 0.5
    kappa = 0.5 + (2.5 - 0.5) * (f_best_clipped - (-40)) / (0 - (-40))
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    lcb = mu - kappa * sigma_safe
    
    # Ensure output is finite
    lcb = np.where(np.isfinite(lcb), lcb, -np.inf)
    
    return lcb
