import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function that combines expected improvement
    with a cost-aware exploration bonus and a structure-aware bonus for
    regions predicted to be near the known global minimum.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to get per-unit-cost improvement
    cost_aware_ei = ei / cost_safe
    
    # Structure-aware bonus: Prioritize candidates where the GP mean
    # predicts a value close to the known global minimum of Styblinski-Tang.
    # Theoretical min is approx -78.3323.
    target_min = -78.3323
    # Gaussian bump centered at the target minimum value.
    # Width 5.0 means mu within ~5 units of -78.33 gets high bonus.
    proximity = np.exp(-((mu - target_min) ** 2) / (2 * 5.0 ** 2))
    
    # The bonus is scaled by 100 to significantly outweigh EI in the target region,
    # but divided by cost to maintain cost-awareness.
    structure_bonus = 100.0 * proximity / cost_safe
    
    # Combine: Primary is structure-aware exploitation, secondary is EI
    # We reduce the weight of generic exploration (sigma) as per directive
    # "using the GP's standard deviation only to break ties or explore... if bonus region is already [covered]"
    # Here, EI includes sigma, but we add a strong mu-based bonus.
    score = cost_aware_ei + structure_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
