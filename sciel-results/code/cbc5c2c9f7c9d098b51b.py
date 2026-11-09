import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    if time.time() > _DEADLINE - 5.0:
        # Fallback: simple cost-normalized LCB
        cost_safe = np.maximum(cost, 1e-6)
        scores = -(mu - 1.0 * sigma) / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate Expected Improvement
    # Z = (mu - f_best) / sigma
    # Clip Z to prevent numerical issues in norm.cdf/pdf
    z = (mu - f_best) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(-z) + sigma * phi(z)
    # Note: Phi(-z) = 1 - Phi(z)
    cdf_neg_z = norm.cdf(-z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_neg_z + sigma_safe * pdf_z
    
    # Cost-Adjusted Score
    # We add a small exploration bonus (sigma) to help escape local minima,
    # scaled by the same cost factor.
    # The weight of the exploration bonus is 0.5 to balance exploitation (EI) and exploration (sigma).
    exploration_bonus = 0.5 * sigma_safe
    
    scores = (ei + exploration_bonus) / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
