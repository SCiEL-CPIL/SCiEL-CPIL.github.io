"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 320  # seconds, safety margin before 330s kill

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Uses a hybrid approach combining Expected Improvement (EI) with an
    adaptive exploration bonus. The exploration component uses a UCB-style
    term with a beta that decays as we get more confident (smaller sigma).
    
    Key structural change: Use a "probability of improvement" weighted
    exploration that naturally balances exploitation and exploration.
    We use a combination of:
    1. Expected Improvement (exploitation)
    2. A variance-weighted exploration term that encourages visiting
       regions with high uncertainty
    
    For Hartmann 6D with only 30 evaluations, we need aggressive exploration
    early and exploitation later. Since we don't have step count, we use
    the magnitude of sigma as a proxy: high sigma regions get more exploration.
    """
    if _time_left() < 5:
        # Fallback: return pure exploit (lowest mu)
        return -mu
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: UCB-style with beta that depends on the
    # relative uncertainty. We want to explore where sigma is large
    # relative to the improvement potential.
    
    # Use a probability-of-improvement based exploration bonus
    # P(I > 0) = Phi((f_best - mu) / sigma)
    # But we want to reward high sigma regions more.
    
    # Structural change: Use a "knowledge gradient" inspired approach
    # where we weight by the probability that this point is the new minimum
    # among all candidates.
    
    # Simpler but effective: use EI + beta * sigma where beta is adaptive
    # beta = 1.0 / (1 + mean(sigma_safe)) to normalize
    # This makes exploration stronger when overall uncertainty is high
    
    # Actually, let's use a more principled approach:
    # The "expected regret" of NOT visiting this point is related to
    # how much it could improve f_best. We use:
    # score = EI + exploration_weight * sigma
    
    # exploration_weight should be higher when we have less information
    # Since we don't have step count, use the fraction of points with
    # high uncertainty as a proxy for "how much we still need to explore"
    
    # Compute a normalized exploration weight
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe) + 1e-10
    
    # Points with high sigma relative to the mean get more exploration bonus
    # This creates a natural diversity incentive
    
    # Use a moderate exploration weight
    # For 25 remaining evaluations, beta ~ 1.5-2.5 works well
    beta = 2.0
    
    # But adapt beta based on the spread of sigma
    # If sigma is very spread out, use higher beta to encourage exploration
    # If sigma is concentrated, use lower beta
    beta_adaptive = beta * (1.0 + 0.5 * (sigma_std / (sigma_mean + 1e-10)))
    
    scores = ei + beta_adaptive * sigma_safe
    
    # Normalize scores to be in a reasonable range
    # This helps avoid numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
