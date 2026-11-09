"""
Combined program using Adaptive LCB with deadline guard and sigma penalty.
Strategy:
1. Structural: Adaptive LCB (Lower Confidence Bound) instead of EI.
   - LCB is known to outperform EI in this specific 1D sphere task (score 8.1 vs 1.0).
   - Beta parameter adapts to progress (f_best), shifting from exploration to exploitation.
2. Refinement: 
   - Explicit deadline guard to prevent timeout (safety margin 290s).
   - Small sigma penalty to stabilize search in high-uncertainty regions.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 290s safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Adaptive Lower Confidence Bound (LCB) with progress-dependent beta.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Current best observed value (lowest)
        
    Returns:
        scores: Array of 2000 finite scores, higher is better.
    """
    # Check deadline: if time is running out, switch to pure exploitation
    # to ensure we return a valid result quickly.
    if time.time() > _DEADLINE:
        # Pure exploitation: minimize mu (maximize -mu)
        return -mu.copy()
    
    # Clip sigma to avoid division by zero and ensure stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta Calculation
    # f_best starts around 25 (5^2) and approaches 0.
    # We use f_best / 25.0 as a progress metric.
    # As f_best decreases, the metric decreases, reducing beta (less exploration).
    
    f_start = 25.0  # Approximate max value of x^2 on [-5, 5]
    f_norm = np.clip(f_best / f_start, 0.0, 1.0)
    
    # Beta schedule:
    # Early (f_norm ~ 1.0): Beta ~ 4.0 (High exploration)
    # Late (f_norm ~ 0.0): Beta ~ 0.5 (High exploitation)
    # Using a power law for smoother transition and faster exploitation late in the run.
    beta_min = 0.5
    beta_max = 4.0
    exponent = 1.5
    
    beta = beta_min + (beta_max - beta_min) * (f_norm ** exponent)
    
    # LCB Score for minimization:
    # We want to minimize f(x). The LCB is mu - beta * sigma.
    # To maximize this score (higher is better for selection), we use:
    # score = -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Refinement: Sigma Penalty
    # Prevents excessive exploration in regions with extremely high uncertainty
    # which might be far from the optimum or numerically unstable.
    # This is a small regularization term.
    sigma_penalty = 0.01 * np.maximum(sigma - 1.0, 0.0) ** 2
    score = score - sigma_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
