"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants
MIN_POSSIBLE = -0.8


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of cosine8 8d.
    
    Structural change: Use a probabilistic "probability of improvement" (PoI)
    metric combined with a cost-aware exploration bonus. PoI is more robust
    than EI in high dimensions as it avoids the multiplicative scaling issues
    of EI when sigma varies widely. We blend PoI with a normalized uncertainty
    term (UCB-like) to balance exploitation and exploration.
    
    The key structural difference from EI/UCB is the use of PoI (a probability
    in [0,1]) which is naturally bounded and less sensitive to outlier sigma values,
    combined with a cost-normalized exploration term.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Probability of Improvement for minimization:
    # PoI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    poi = norm.cdf(z)
    
    # Cost-adjusted PoI: reward improvement probability per unit cost
    poi_cost = poi / cost_safe

    # Exploration term: normalized uncertainty
    # Use sigma directly as exploration signal, normalized by cost
    exploration = sigma_safe / cost_safe

    # Adaptive blending based on progress toward the minimum
    # progress: 0 at optimum (-0.8), 1 at worst expected start (~0.0)
    progress = np.clip((f_best - MIN_POSSIBLE) / (0.0 - MIN_POSSIBLE), 0.0, 1.0)
    
    # Exploration weight: higher when far from optimum
    # Use a smoother, more aggressive exploration early
    w_explore = 0.1 + 0.5 * progress**1.5  # ranges from 0.1 to 0.6
    w_exploit = 1.0 - w_explore

    # Normalize both components to zero mean, unit std for comparable scales
    poi_mean = np.mean(poi_cost)
    poi_std = np.std(poi_cost) + 1e-12
    poi_norm = (poi_cost - poi_mean) / poi_std

    expl_mean = np.mean(exploration)
    expl_std = np.std(exploration) + 1e-12
    expl_norm = (exploration - expl_mean) / expl_std

    # Combine
    score = w_exploit * poi_norm + w_explore * expl_norm

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
