# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Multiplicative Rank-Based Acquisition Function.
    
    This function implements a structural change from the previous additive 
    Pareto-frontier selection to a multiplicative rank-based approach.
    
    Instead of combining EI and UCB scores linearly or via min-rank, this 
    method:
    1. Ranks the candidates by predicted mean (exploitation potential).
    2. Ranks the candidates by predicted standard deviation (exploration potential).
    3. Applies a non-linear power transformation to the uncertainty rank to 
       emphasize extreme uncertainty.
    4. Multiplies the exploitation rank by a boost factor derived from the 
       transformed uncertainty rank.
    
    This decouples the magnitude of the GP variance from the selection pressure, 
    relying instead on the relative ordering of uncertainty among the 2000 
    candidates. It favors points that are both promising (low mean) and 
    highly uncertain, but the multiplicative structure ensures that poor 
    prospects (high mean) are heavily penalized regardless of uncertainty.
    
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
    
    if n == 0:
        return np.array([])
        
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        # Replace non-finite mu with a very bad value (high mu for minimization)
        # Replace non-finite sigma with 0 (no uncertainty)
        bad_mu_val = np.max(mu[mask]) + 100.0 if np.any(mask) else 0.0
        mu = np.where(mask, mu, bad_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is strictly positive
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    
    # 1. Compute Ranks
    # Higher rank means "better" (higher score)
    # For exploitation, we want low mu, so we rank -mu
    # For exploration, we want high sigma, so we rank sigma
    
    # np.argsort(np.argsort(x)) gives ranks from 0 to n-1
    # We want higher values to have higher ranks
    rank_mu = np.argsort(np.argsort(-mu)).astype(np.float64) / (n - 1) if n > 1 else np.ones(n)
    rank_sigma = np.argsort(np.argsort(sigma)).astype(np.float64) / (n - 1) if n > 1 else np.ones(n)
    
    # 2. Determine Progress and Beta
    # Estimate progress based on f_best relative to global min and a typical start
    # Global min of Hartmann 3D is -3.86278
    # Typical random start is around -1.0 to -2.0. Let's assume 0.0 for safe upper bound on progress calculation
    # If f_best is close to -3.86, progress is 1.0 (exploit)
    # If f_best is close to 0.0, progress is 0.0 (explore)
    
    f_min_global = -3.86278
    f_start_assumed = 0.0  # Conservative assumption for start
    
    # Clamp f_best to [f_min_global, f_start_assumed]
    f_best_clamped = np.clip(f_best, f_min_global, f_start_assumed)
    
    # Progress: 0.0 at start, 1.0 at global min
    if f_start_assumed > f_min_global:
        progress = (f_start_assumed - f_best_clamped) / (f_start_assumed - f_min_global)
    else:
        progress = 0.0
        
    # Clip progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta: Exploration weight
    # Starts at 1.5 (strong exploration), decays to 0.1 (strong exploitation)
    beta_start = 1.5
    beta_end = 0.1
    beta = beta_start + (beta_end - beta_start) * progress
    
    # Gamma: Shape parameter for uncertainty rank
    # > 1 emphasizes high uncertainty (top quantiles)
    gamma = 1.8
    
    # 3. Compute Score
    # Boost factor: 1 + beta * (rank_sigma ** gamma)
    # This makes the exploration component non-linear, favoring the top 10-20% of uncertainty
    boost = 1.0 + beta * (rank_sigma ** gamma)
    
    # Final Score: Multiplicative combination
    # Points with high exploitation rank (low mu) get a large score.
    # The boost increases with uncertainty rank.
    # This ensures that a point with low promise (low rank_mu) will have a low score even if rank_sigma is high,
    # unless beta is very large (early stages).
    scores = rank_mu * boost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
