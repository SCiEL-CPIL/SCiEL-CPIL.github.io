"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Pure GP-LCB acquisition function.
    
    For minimization, the Lower Confidence Bound is:
    LCB(x) = mu(x) - kappa * sigma(x)
    
    We want to minimize the objective, so we want to maximize the score
    such that higher score means "more promising to minimize".
    Since LCB is the lower bound on the function value, a lower LCB is better.
    Therefore, we return -LCB(x) = -mu(x) + kappa * sigma(x).
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float
    
    Returns:
    score: array of shape (2000,), finite values, higher is better.
    """
    # Ensure sigma is non-negative and avoid division issues if any (though not dividing here)
    # sigma is standard deviation, should be >= 0.
    # If sigma is very small, kappa*sigma is small, score is dominated by -mu.
    # If sigma is large, score is dominated by kappa*sigma (exploration).
    
    # Fixed kappa for pure LCB.
    # Typical values for kappa in BO are 1.0 to 2.0.
    # Given the task asks for a structural change to pure LCB, we pick a standard value.
    # kappa = 2.0 is a common choice for balancing exploration/exploitation.
    kappa = 2.0
    
    # Calculate LCB score: -mu + kappa * sigma
    score = -mu + kappa * sigma
    
    # Ensure finite output
    # mu and sigma should be finite from the GP, but just in case
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
