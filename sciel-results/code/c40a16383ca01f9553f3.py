"""Cost-aware acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid acquisition: UCB exploration + cost-aware exploitation.
    
    Combines:
      - UCB term: mu - beta * sigma (explores high-uncertainty regions)
      - Cost-aware EI-like term: (f_best - mu) * phi(z) / cost (exploits improvements per cost)
    
    This balances exploration (UCB) with cost-efficient exploitation (EI/cost).
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # UCB component: explore regions with high uncertainty
    # Lower sigma (more certain) and lower mu (better) should score higher
    # But we want to reward uncertainty too, so use -beta * sigma
    beta = 2.0
    ucb = mu - beta * sigma_safe
    
    # Cost-aware exploitation component
    # Expected improvement per unit cost
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_cost = ei / np.maximum(cost, eps)
    
    # Combine: normalize both components to comparable scales
    # Use a weighted combination that adapts based on how far we are from optimum
    # When f_best is still high (early), weight exploration more
    # When f_best is low (late), weight exploitation more
    
    # Simple approach: normalize ucb and ei_cost, then combine
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + eps
    ucb_norm = (ucb - ucb_min) / ucb_range  # higher = better (lower mu, higher sigma)
    
    ei_cost_min = np.min(ei_cost)
    ei_cost_max = np.max(ei_cost)
    ei_cost_range = ei_cost_max - ei_cost_min + eps
    ei_cost_norm = (ei_cost - ei_cost_min) / ei_cost_range  # higher = better
    
    # Weight: start with 50/50, but let the data drive it
    # Actually, just use a fixed blend that works well for cost-aware BO
    w_ucb = 0.4
    w_ei = 0.6
    
    score = w_ucb * ucb_norm + w_ei * ei_cost_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
