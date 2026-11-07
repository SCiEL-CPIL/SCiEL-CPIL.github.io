"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) with a UCB-like
    exploration term and a safety guard against numerical issues.
    
    We use a weighted combination:
    score = w_ei * EI + w_ucb * UCB
    
    EI is the standard expected improvement for minimization.
    UCB is an upper confidence bound: mu - kappa * sigma (for minimization, we want low mu and high sigma).
    
    We also clip sigma to avoid division by zero or extreme values.
    """
    # Ensure sigma is positive and not too small
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But we need to handle the case where f_best - mu is negative (mu < f_best)
    # In that case, EI should be positive (we expect improvement)
    # Actually, EI formula is:
    # If mu < f_best: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If mu >= f_best: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # The formula is the same, but z will be negative when mu < f_best.
    
    # Clip z to avoid numerical issues in cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # UCB for minimization: we want to maximize mu - kappa * sigma
    # This favors points with low mean and high uncertainty
    kappa = 2.0  # Exploration-exploitation balance
    ucb = mu - kappa * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is also in objective units
    # But they have different scales. Let's use a weighted sum.
    # We can normalize by the range of each, or just use a fixed weight.
    
    # Let's use a simple weighted sum with tunable weights
    # w_ei = 1.0, w_ucb = 0.5
    # But we need to be careful about the scale.
    
    # Alternative: Use EI only, but with a better formulation
    # Or use a hybrid that adapts based on the number of evaluations
    
    # For now, let's try a simple hybrid:
    # score = ei + 0.5 * (f_best - ucb)  # This doesn't make sense dimensionally
    
    # Better: Just use EI, but with a safety margin
    # Or use UCB directly
    
    # Let's try: score = ei + epsilon * (1 / (sigma + epsilon))
    # This adds a small bonus for high uncertainty
    
    # Actually, let's stick with EI but add a small exploration bonus
    # The standard EI already balances exploration and exploitation
    
    # Let's try a different approach: Use the Probability of Improvement (POI)
    # POI = Phi(z) = Phi((f_best - mu) / sigma)
    # This is simpler and sometimes works better
    
    # Or use a combination: score = w1 * EI + w2 * POI
    
    # Let's try: score = EI + 0.1 * POI
    poi = norm.cdf(z_clipped)
    
    score = ei + 0.1 * poi
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
