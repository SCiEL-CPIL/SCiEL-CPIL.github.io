import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill
_total_evals = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid EI + UCB approach with adaptive blending.
    Early iterations favor exploration (higher UCB weight),
    later iterations favor exploitation (higher EI weight).
    """
    global _total_evals
    _total_evals += 1
    
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Adaptive blending based on evaluation count
    # Total budget is 30 evaluations per seed
    # Early: more UCB (exploration), Late: more EI (exploitation)
    progress = min(_total_evals / 30.0, 1.0)
    
    # Blend weight: early=0.7 UCB, late=0.3 UCB
    ucb_weight = 0.3 + 0.4 * (1.0 - progress)
    ei_weight = 1.0 - ucb_weight
    
    # UCB score for minimization
    beta = 1.5 + 2.0 * np.clip(f_best, 0.0, 1.0)
    ucb_score = -mu + beta * sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    ei_score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize scores to [0, 1] range for stable blending
    def normalize(s):
        s_min, s_max = np.min(s), np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ucb_norm = normalize(ucb_score)
    ei_norm = normalize(ei_score)
    
    # Blend
    score = ucb_weight * ucb_norm + ei_weight * ei_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
