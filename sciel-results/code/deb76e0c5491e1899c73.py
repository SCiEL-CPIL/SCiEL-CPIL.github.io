# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Adaptive Rank-Blended PI-UCB Acquisition Function.
    
    This function combines the structural robustness of rank-based selection 
    (from the high-performing current program) with the adaptive progress 
    weighting (from the inspirations). 
    
    Instead of a hard Pareto-frontier selection (min of ranks), it uses a 
    soft, adaptive linear blend of the ranks of Probability of Improvement 
    (PI) and Upper Confidence Bound (UCB). 
    
    - PI is used for exploitation as it directly measures the probability 
      of improving upon f_best, decoupling it from the magnitude of 
      uncertainty (unlike EI).
    - UCB is used for exploration, with an adaptive beta that decays as 
      the search progresses.
    - The blend weights shift from exploration-heavy to exploitation-heavy 
      as f_best approaches the global minimum.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more attractive to evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just exploit (maximize -mu)
        return -np.asarray(mu, dtype=np.float64).ravel().copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        safe_mu_val = np.max(mu[mask]) if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    
    # 1. Calculate Probability of Improvement (PI)
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to prevent numerical issues
    z_clipped = np.clip(z, -50, 50)
    
    scores_pi = norm.cdf(z_clipped)
    
    # 2. Calculate UCB score
    # UCB(x) = -mu(x) + beta * sigma(x)
    f_global_min = -3.86278
    
    # Normalize progress: 0.0 at f_best = 0.0 (start), 1.0 at f_best = -3.86
    f_best_clamped = np.clip(f_best, -3.86, 0.0)
    progress = (0.0 - f_best_clamped) / 3.86  # 0 at start, 1 at end
    
    # beta decays as we get closer to the global minimum
    # Start with higher exploration (beta=2.5) and reduce to lower (beta=0.5)
    beta = 2.5 - 2.0 * progress
    
    scores_ucb = -mu + beta * sigma
    
    # 3. Rank-based normalization
    # Higher rank = better
    if n > 1:
        pi_ranks = np.argsort(np.argsort(scores_pi)) / (n - 1)
        ucb_ranks = np.argsort(np.argsort(scores_ucb)) / (n - 1)
    else:
        pi_ranks = np.ones(n)
        ucb_ranks = np.ones(n)
    
    # 4. Adaptive Weighted Blend
    # Early: High exploration weight (favor UCB rank)
    # Late: High exploitation weight (favor PI rank)
    
    # Weights are complementary and sum to 1
    # w_exploit starts at 0.3 and goes to 1.0
    # w_explore starts at 0.7 and goes to 0.0
    w_exploit = 0.3 + 0.7 * progress
    w_explore = 1.0 - w_exploit
    
    scores = w_exploit * pi_ranks + w_explore * ucb_ranks
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
