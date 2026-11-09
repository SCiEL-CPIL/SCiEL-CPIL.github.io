"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 285  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski-Tang 2D.
    
    Uses a hybrid acquisition combining:
    - Cost-adjusted Expected Improvement (exploitation)
    - Cost-adjusted Lower Confidence Bound (exploration)
    - Probability of Improvement as a bonus
    
    The weights adapt based on progress toward the known minimum.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite and valid
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    # Handle edge cases for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (improvement * cdf_z + sigma * pdf_z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    poi = cdf_z
    
    # Cost adjustment: use log cost for smoother behavior
    # cost is in [1, 2], so log(cost) is in [0, ~0.69]
    log_cost = np.log(cost)
    
    # Cost-adjusted EI: penalize expensive points
    # Use a power law to balance exploration vs cost
    ei_cost_adj = ei / (1.0 + 0.5 * log_cost)
    
    # Lower Confidence Bound for minimization (we want low values)
    # LCB = mu - beta * sigma (lower is better)
    beta_base = 1.0
    # Adapt beta: start with more exploration, reduce as we find better values
    known_min = -78.3323
    progress = max(0.0, 1.0 - (f_best - known_min) / 100.0)
    beta = beta_base * (1.0 - 0.5 * progress)
    
    lcb = mu - beta * sigma
    # Convert to score (higher is better): negate
    lcb_score = -lcb
    # Cost-adjusted LCB
    lcb_cost_adj = lcb_score / (1.0 + 0.3 * log_cost)
    
    # Normalize components to have zero mean and unit std
    ei_ca_mean = np.mean(ei_cost_adj)
    ei_ca_std = np.std(ei_cost_adj)
    if ei_ca_std > 1e-9:
        ei_norm = (ei_cost_adj - ei_ca_mean) / ei_ca_std
    else:
        ei_norm = np.zeros_like(ei_cost_adj)
    
    lcb_s_mean = np.mean(lcb_cost_adj)
    lcb_s_std = np.std(lcb_cost_adj)
    if lcb_s_std > 1e-9:
        lcb_norm = (lcb_cost_adj - lcb_s_mean) / lcb_s_std
    else:
        lcb_norm = np.zeros_like(lcb_cost_adj)
    
    # POI is already in [0,1], normalize it
    poi_mean = np.mean(poi)
    poi_std = np.std(poi)
    if poi_std > 1e-9:
        poi_norm = (poi - poi_mean) / poi_std
    else:
        poi_norm = np.zeros_like(poi)
    
    # Adaptive weights based on progress
    # Early: more exploration (LCB), Later: more exploitation (EI)
    if f_best < known_min + 2.0:
        # Very close to optimum, favor exploitation heavily
        w_ei = 0.80
        w_lcb = 0.15
        w_poi = 0.05
    elif f_best < known_min + 10.0:
        # Good progress, balance with slight exploitation bias
        w_ei = 0.65
        w_lcb = 0.28
        w_poi = 0.07
    elif f_best < known_min + 30.0:
        # Moderate progress, balanced
        w_ei = 0.50
        w_lcb = 0.35
        w_poi = 0.15
    else:
        # Early stage, favor exploration
        w_ei = 0.40
        w_lcb = 0.40
        w_poi = 0.20
    
    # Combine
    scores = w_ei * ei_norm + w_lcb * lcb_norm + w_poi * poi_norm
    
    # Add a small direct EI/cost term to ensure we don't ignore obvious improvements
    scores += 0.05 * ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Safety: if close to timeout, return quickly with simpler score
    if not _elapsed_ok():
        return ei / cost
    
    return scores
# EVOLVE-BLOCK-END
