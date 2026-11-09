"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of the Beale 2D function.
    
    Uses a hybrid strategy combining:
    1. UCB (Upper Confidence Bound) for exploration
    2. Expected Improvement (EI) for exploitation
    3. A diversity-promoting term based on uncertainty
    
    The Beale function has a narrow valley, so we need both exploration
    and exploitation. We use a weighted combination that adapts based on
    the current state of optimization.
    """
    # Ensure sigma is positive and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute UCB score (exploration)
    # Beta controls the exploration-exploitation tradeoff
    # Start with moderate exploration, could be adapted
    beta = 2.5
    ucb = mu - beta * sigma
    
    # Compute Expected Improvement (exploitation)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to be on a comparable scale with UCB
    # EI can be zero or negative, UCB can be any value
    # We want EI to contribute when it's positive (improvement expected)
    
    # Compute a diversity/uncertainty bonus
    # Points with high uncertainty are worth exploring
    uncertainty_bonus = sigma
    
    # Weighted combination
    # We want to balance:
    # - Exploitation (EI): high when improvement is likely
    # - Exploration (UCB): high when uncertainty is high
    # - Diversity: encourages exploring new regions
    
    # Adaptive weighting based on how good f_best is relative to the function range
    # Beale function minimum is 0, maximum in [-4.5, 4.5]^2 is around 230
    # If f_best is still high, we should explore more
    # If f_best is close to 0, we should exploit more
    
    # Simple heuristic: weight based on normalized f_best
    # f_best starts around 100-200, should decrease to ~0
    # Use a sigmoid-like weighting
    f_best_normalized = np.clip(f_best / 50.0, 0, 1)  # 0 when f_best <= 0, 1 when f_best >= 50
    
    # When f_best is high (poor), weight exploration more
    # When f_best is low (good), weight exploitation more
    w_explore = 0.3 + 0.4 * f_best_normalized  # 0.3 to 0.7
    w_exploit = 0.7 - 0.4 * f_best_normalized  # 0.7 to 0.3
    
    # Ensure weights sum to 1 and are positive
    w_explore = max(w_explore, 0.1)
    w_exploit = max(w_exploit, 0.1)
    w_total = w_explore + w_exploit
    w_explore /= w_total
    w_exploit /= w_total
    
    # Combine scores
    # Scale EI to be comparable with UCB
    # EI is typically in [0, f_best], UCB is in [min_mu - beta*max_sigma, max_mu]
    # We'll normalize EI by dividing by (f_best + epsilon) to get a [0, 1] scale
    ei_normalized = ei / (f_best + 1e-9)
    ei_normalized = np.clip(ei_normalized, 0, 1)
    
    # UCB is already in objective units, we want to maximize it
    # But we need to be careful about the scale
    # Let's use a combined score that's in a consistent scale
    
    # Final score: higher is better
    score = w_exploit * ei + w_explore * (f_best - ucb) + 0.1 * uncertainty_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
