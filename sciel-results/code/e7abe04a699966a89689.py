"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = None
_DEADLINE = None

def _ensure_deadline():
    global _START_TIME, _DEADLINE
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 320.0  # 320s safety margin within 330s limit
    return _DEADLINE


def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    
    # Hybrid acquisition: blend UCB (exploration) and EI (exploitation)
    # UCB: -mu + kappa * sigma
    # EI: (f_best - mu) * cdf(z) + sigma * pdf(z)
    
    # Compute EI scores
    z = (f_best - mu) / (sigma + 1e-9)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB scores with high exploration coefficient
    kappa = 3.0
    ucb_scores = -mu + kappa * sigma
    
    # Normalize both score sets to [0, 1] range for blending
    # Use robust normalization to handle outliers
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ei_norm = normalize_scores(ei_scores)
    ucb_norm = normalize_scores(ucb_scores)
    
    # Blend: weight towards UCB for exploration, EI for exploitation
    # As f_best improves, we might want to shift more towards exploitation
    # But with only 30 evaluations, exploration is critical
    alpha = 0.6  # 60% UCB, 40% EI
    scores = alpha * ucb_norm + (1 - alpha) * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
