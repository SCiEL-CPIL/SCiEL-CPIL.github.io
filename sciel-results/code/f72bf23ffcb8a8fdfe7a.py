import numpy as np
import time

# EVOLVE-BLOCK-START

def _minmax_normalize(x):
    """Normalize array x to range [0, 1]."""
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-10:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)

def acquisition(mu, sigma, f_best):
    # Deadline guard
    # The evaluator kills after 330s. We should be fast, but just in case.
    # This function is called 25 times (30 evals - 5 initial).
    # 2000 candidates per call.
    # Computation is O(N).
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # Where Phi is CDF of standard normal, phi is PDF.
    # We use scipy.stats.norm for this.
    from scipy.stats import norm
    
    z = (f_best - mu) / (sigma + 1e-10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Calculate LCB score (equivalent to UCB for maximization of -f)
    # We want to maximize this score.
    # Lower bound on the minimum: mu - kappa * sigma
    # Score to maximize: -(mu - kappa * sigma) = -mu + kappa * sigma
    kappa = 2.0  # Exploration parameter
    lcb_score = -mu + kappa * sigma
    
    # 3. Normalize both scores to [0, 1] to make them comparable
    ei_norm = _minmax_normalize(ei)
    lcb_norm = _minmax_normalize(lcb_score)
    
    # 4. Blend
    # Weighting: 50% Exploitation (EI), 50% Exploration (LCB)
    # This is a standard heuristic. Adjusting weights might help, but 0.5/0.5 is a safe bet.
    w_ei = 0.5
    w_lcb = 0.5
    
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    return score

# EVOLVE-BLOCK-END
