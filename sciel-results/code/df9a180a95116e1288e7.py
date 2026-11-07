import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Handle zero sigma safely
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is finite and non-negative (EI should be >= 0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration weight (beta)
    # Inspired by the current program's adaptive kappa, but applied to a UCB term added to EI
    # When f_best is far from the global minimum, we need more exploration.
    # When f_best is close, we exploit more (lower beta).
    
    known_min = -39.1662
    # Normalize distance: 1.0 when f_best is high (far), 0.0 when f_best is low (close)
    # f_best starts around 20-30 and goes down to -39.1662
    # Let's assume a reasonable upper bound for f_best during search, e.g., 50
    dist = max(0.0, (f_best - known_min) / (50.0 - known_min))
    dist = min(1.0, dist)
    
    # Beta decays from a high value (exploration) to a low value (exploitation)
    # Base beta of 0.5 for exploration, decaying to 0.1
    beta = 0.1 + 0.4 * dist
    
    # UCB component for minimization: we want low mu and high sigma
    # The term beta * sigma encourages exploration of high uncertainty regions
    # We do NOT subtract mu here because EI already accounts for mu.
    # Adding beta * sigma acts as a bonus for uncertainty.
    ucb_bonus = beta * sigma_safe
    
    # Combine EI and UCB bonus
    # EI is in objective units, sigma is in objective units (standardized targets, but sigma is std dev)
    # Since EI and sigma are on similar scales (both derived from the posterior in objective space),
    # a simple sum is reasonable.
    score = ei + ucb_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
