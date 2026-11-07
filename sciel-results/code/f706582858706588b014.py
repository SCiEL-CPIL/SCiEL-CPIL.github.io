import numpy as np
import time

# Constants
_EPS = 1e-12
_DEADLINE_MARGIN = 280.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition function for Bayesian optimization of sphere 1d.
    
    Strategy:
    - Use Upper Confidence Bound (UCB) for minimization: score = mu - kappa * sigma
    - Adaptively adjust kappa based on f_best.
    - Early in optimization (f_best is large), use high kappa to explore.
    - Late in optimization (f_best is small), use low kappa to exploit.
    
    Parameters:
    mu: array of shape (2000,) with mean predictions
    sigma: array of shape (2000,) with standard deviation predictions
    f_best: scalar, the lowest observed value so far
    
    Returns:
    scores: array of shape (2000,) with higher values meaning more promising
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and avoid division by zero issues
    sigma = np.maximum(sigma, _EPS)
    
    # Adaptive kappa selection
    # f_best for sphere 1d on [-5, 5] ranges from ~50 (worst) to 0 (best).
    # We want kappa to be large when f_best is large, and small when f_best is small.
    # A simple linear mapping or exponential decay can work.
    # Let's use a mapping that transitions from high exploration to low.
    # 
    # If f_best > 10, we are likely still exploring. kappa ~ 5-10.
    # If f_best < 1, we are likely exploiting. kappa ~ 0.5-1.
    #
    # Let's use a smooth transition:
    # kappa = base_kappa * (f_best / scale_factor) + min_kappa
    # Or simply: kappa = max(min_kappa, min(max_kappa, f_best * factor))
    
    # Given the range of f_best (0 to ~50), let's try:
    # kappa = 1.0 + 2.0 * f_best  (linear increase with f_best)
    # If f_best = 0, kappa = 1.0 (exploit)
    # If f_best = 1, kappa = 3.0
    # If f_best = 5, kappa = 11.0 (explore)
    # If f_best = 10, kappa = 21.0 (very explore)
    
    # Let's cap it to avoid extreme values if f_best is very high initially.
    # Initial points are uniform on [-5, 5], so f_best could be up to 25 or 50.
    # Let's cap kappa at 10.0.
    
    kappa = 1.0 + 2.0 * f_best
    kappa = min(kappa, 10.0)
    kappa = max(kappa, 0.5) # Ensure some exploration always
    
    # Compute UCB scores for minimization
    # We want to find low values, so we prefer low mu and high sigma.
    # Score = -mu + kappa * sigma  (Higher score = better)
    # Or equivalently: score = -(mu - kappa * sigma)
    
    scores = -mu + kappa * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time budget (though this is very fast)
    if time.time() - start_time > _DEADLINE_MARGIN:
         # Fallback if somehow slow (shouldn't happen)
         return np.zeros_like(mu)
         
    return scores
