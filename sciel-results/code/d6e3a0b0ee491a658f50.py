import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    Returns scores where higher values indicate better candidates to evaluate.
    """
    # Beta controls the trade-off between exploitation (low mu) and exploration (high sigma).
    # A higher beta favors exploration, which helps in multimodal functions like Hartmann 3D.
    beta = 2.0
    
    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division issues if we were using z-scores, 
    # though here we just multiply.
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: mu - beta * sigma
    # We want to MAXIMIZE the score, so we return the negative of LCB:
    # Score = -mu + beta * sigma
    
    score = -mu + beta * sigma
    
    # Replace any non-finite values with a very low score to avoid crashing the evaluator
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
