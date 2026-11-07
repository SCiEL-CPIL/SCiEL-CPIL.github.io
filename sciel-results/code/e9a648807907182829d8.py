"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, keep 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at 2000 candidate points
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points
    f_best : float
        Best observed function value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.ones(2000) / (np.maximum(cost, 1e-9) + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -50.0, 50.0)
    
    # Expected Improvement (EI) - for minimization, improvement is f_best - mu
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    ei_cost = ei / cost
    
    # LCB (Lower Confidence Bound) for minimization: mu - beta * sigma
    # Higher LCB score means more likely to be better
    # Dynamic beta: start with exploration, reduce as budget depletes
    # We don't know total budget exactly, but we can use a heuristic
    # Based on the problem: total cost budget is 30, initial 5 points cost ~2.5
    # So remaining budget is roughly 30 - current_cost, but we don't track that here.
    # Use a fixed beta that balances exploration and exploitation well in 8D.
    beta = 1.5
    lcb_score = -(mu - beta * sigma)  # Negate because we want higher = better
    lcb_score = lcb_score / cost
    
    # Probability of Improvement (PI) - for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z_clipped)
    pi_cost = pi / cost
    
    # Blend: primarily use cost-normalized EI, with some LCB and PI
    # The key insight from the knowledge: blending PoI with EI helps robustness
    # Also add a soft uncertainty-reduction component
    
    # Soft uncertainty bonus: rewards regions with moderate uncertainty
    # This steers toward under-explored regions without pure exploitation
    # Use a sigmoid-like function that peaks at moderate sigma
    sigma_mean = np.mean(sigma)
    # Normalize sigma to [0, 1] roughly
    sigma_norm = sigma / (sigma_mean + 1e-9)
    # Bonus peaks around sigma_norm ~ 0.5-1.0
    uncertainty_bonus = np.exp(-0.5 * (sigma_norm - 0.8) ** 2)
    uncertainty_bonus = uncertainty_bonus / cost
    
    # Combine all components
    # Weights: EI is primary, LCB for exploration, PI for robustness, uncertainty for diversity
    w_ei = 1.0
    w_lcb = 0.3
    w_pi = 0.2
    w_unc = 0.1
    
    scores = w_ei * ei_cost + w_lcb * lcb_score + w_pi * pi_cost + w_unc * uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
