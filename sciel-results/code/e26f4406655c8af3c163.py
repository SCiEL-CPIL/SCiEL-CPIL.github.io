"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to avoid timeout
_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI/UCB acquisition for Bayesian optimization of Ackley 2D.
    
    Combines Expected Improvement (exploitation) with UCB (exploration)
    using an adaptive weight that shifts from exploration to exploitation
    as f_best approaches the minimum value of 0.
    
    Structural change: Uses median-normalized sigma in the UCB term to stabilize
    exploration relative to the typical uncertainty magnitude of the current GP
    posterior, preventing regions with naturally higher/lower GP variance from
    being overly favored or penalized solely due to scale.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value.
    
    Returns 2000 finite scores, higher means evaluate next.
    """
    # Deadline guard: return a simple fallback if running out of time
    if _time_check():
        return 0.5 * sigma - mu
    
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Compute Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Structural Change: Normalize sigma by its median for the UCB term.
    # This makes the exploration term 'beta * sigma_norm' relative to the
    # typical uncertainty level of the current GP posterior, preventing
    # regions with naturally higher/lower GP variance (due to kernel
    # effects or data sparsity) from being overly favored or penalized
    # solely due to scale.
    med_sigma = np.median(sigma)
    if med_sigma < 1e-12:
        med_sigma = 1e-12
    sigma_norm = sigma / med_sigma
    
    # Adaptive Beta:
    # We want to explore more when f_best is high (early) and exploit
    # more when f_best is low (late).
    # Using a logarithmic scale for f_best to handle the wide range of
    # Ackley values (0 to ~20) more gracefully than linear clipping.
    # log(1 + f_best) is 0 at f_best=0, ~3 at f_best=20.
    # We map this to a beta range.
    # Calibrated:
    # beta ~ 2.0 at f_best=5 (early), ~ 0.5 at f_best=0 (late)
    # beta = 0.5 + 1.5 * (log(1+f_best) / log(1+5))
    
    f_best_safe = max(f_best, 0.0)
    ratio = np.log(1.0 + f_best_safe) / np.log(1.0 + 5.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    
    beta = 0.5 + 1.5 * ratio
    
    # Compute UCB with normalized sigma
    UCB = -mu + beta * sigma_norm
    
    # Adaptive weight for combining EI and UCB:
    # When f_best is close to 0, rely more on EI (exploitation)
    # When f_best is large, rely more on UCB (exploration)
    
    if f_best < 0.1:
        w_ucb = 0.2  # More exploitation
    elif f_best < 0.5:
        w_ucb = 0.4
    elif f_best < 2.0:
        w_ucb = 0.6
    else:
        w_ucb = 0.8  # More exploration early on
    
    # Combine: score = w_ucb * UCB + (1 - w_ucb) * EI
    scores = w_ucb * UCB + (1 - w_ucb) * EI
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
