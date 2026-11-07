import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive Upper Confidence Bound (UCB) for minimization.
    # We want to minimize f, so we maximize -mu + beta * sigma.
    # beta is increased when f_best is large (poor) to encourage exploration.
    # beta is decreased when f_best is small (good) to encourage exploitation.
    
    # Clamp f_best to avoid extreme beta values if f_best is very large or negative (though f>=0 here)
    fb = max(0.0, f_best)
    
    # Start with a base beta of 1.0.
    # Add up to 4.0 extra exploration as f_best increases.
    # This creates a strong exploration bias initially.
    beta = 1.0 + min(4.0, fb)
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # If sigma or mu have NaN/Inf, handle them?
    # The prompt says "Return 2000 finite scores".
    # GP outputs should be finite, but just in case:
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
