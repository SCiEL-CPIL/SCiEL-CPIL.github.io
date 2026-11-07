# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute an acquisition score for each candidate.
    Returns a (2000,) array of finite scores (higher = more preferred).
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return simple EI to avoid crash
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Lower Confidence Bound (LCB) based score
    # For minimization: LCB = mu - beta * sigma, we want low LCB
    # Score = -LCB = -mu + beta * sigma
    # Use beta=1.0 for moderate exploration (less aggressive than 1.5)
    beta = 1.0
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Normalize both objectives to comparable scales
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < eps:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range < eps:
        ucb_norm = np.ones_like(ucb_score)
    else:
        ucb_norm = (ucb_score - ucb_min) / ucb_range

    # 4. Combine EI and UCB with a weighted sum
    # EI is primary; UCB provides exploration
    w_ei = 0.8
    w_ucb = 0.2
    
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # 5. Compute Cost Efficiency
    # Cost is in [1, 2]. Use 1/cost as efficiency measure.
    cost_eff = 1.0 / (cost + eps)
    
    cost_min = np.min(cost_eff)
    cost_max = np.max(cost_eff)
    cost_range = cost_max - cost_min
    if cost_range < eps:
        cost_norm = np.ones_like(cost_eff)
    else:
        cost_norm = (cost_eff - cost_min) / cost_range

    # 6. Soft Pareto scoring: instead of hard Pareto frontier,
    # use a smooth combination that rewards high combined score AND high cost efficiency
    # This avoids the binary 0/10 split and allows gradient-based selection
    
    # Method: weighted geometric mean or product of normalized scores
    # score = combined^alpha * cost_norm^beta
    # This naturally balances both objectives smoothly
    
    alpha = 1.0
    beta_cost = 0.5  # Less weight on cost to prioritize improvement
    
    # Add small epsilon to avoid log(0) issues
    combined_safe = np.maximum(combined, eps)
    cost_safe = np.maximum(cost_norm, eps)
    
    # Log-space combination for stability
    log_combined = np.log(combined_safe)
    log_cost = np.log(cost_safe)
    
    log_score = alpha * log_combined + beta_cost * log_cost
    
    # Convert back to linear scale
    soft_pareto_score = np.exp(log_score)
    
    # Normalize to [0, 1] for consistent scaling
    sp_min = np.min(soft_pareto_score)
    sp_max = np.max(soft_pareto_score)
    sp_range = sp_max - sp_min
    if sp_range < eps:
        final_scores = np.ones_like(soft_pareto_score)
    else:
        final_scores = (soft_pareto_score - sp_min) / sp_range
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores

# EVOLVE-BLOCK-END
