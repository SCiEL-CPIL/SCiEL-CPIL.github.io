"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track number of evaluations
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: if we're running long, just return something safe
    # (though with 2000 points and simple math, this shouldn't be an issue)
    
    # Quantile-based acquisition function (5th percentile)
    # This is equivalent to LCB with beta = 1.645
    # score = -(mu - 1.645 * sigma) = -mu + 1.645 * sigma
    
    # However, we can combine this with a slight adaptive component
    # Early on, use a slightly higher quantile (more exploration)
    # Later, use a lower quantile (more exploitation)
    
    # The 5th percentile corresponds to z = 1.645
    # Let's use a fixed quantile of 5th percentile for robustness
    # This directly targets the lower tail of the posterior distribution
    
    quantile_z = 1.645  # 5th percentile
    
    # Score: higher means more promising (lower predicted value + uncertainty)
    # For minimization, we want low mu and high sigma (exploration)
    # score = -mu + quantile_z * sigma
    
    scores = -mu + quantile_z * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
