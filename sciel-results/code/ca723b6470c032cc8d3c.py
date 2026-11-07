# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for cost-aware Bayesian optimization on Shekel 4d.
    
    Uses a modified Expected Improvement (EI) that balances the expected
    improvement against the evaluation cost. The key insight is that since
    cost is proportional to 1 + mean(x_scaled), we should prefer points with
    both high expected improvement AND low cost. We use a log-space formulation
    to make the cost penalty more effective, combined with a floor on sigma to
    avoid numerical issues.
    
    We use: score = EI(x) / (cost(x))^alpha, where alpha > 1 to penalize cost
    more aggressively. Additionally, we add a small exploration bonus based on
    the uncertainty to encourage exploration in early iterations.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute standard normal CDF and PDF for the improvement
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: divide by cost raised to a power > 1
    # This penalizes expensive evaluations more heavily
    # Use a moderate exponent to balance exploration and exploitation
    alpha = 1.5
    cost_safe = np.maximum(cost, 1e-8)
    
    # Base score: EI per unit cost^alpha
    score = ei / (cost_safe ** alpha)
    
    # Add a small exploration bonus: prefer points with higher uncertainty
    # when the expected improvement is low (early in optimization)
    # This helps escape local optima in the acquisition landscape
    # The bonus is scaled down as optimization progresses (implicitly via EI)
    exploration_bonus = sigma_safe * 0.1 / cost_safe
    
    # Combine: the exploration bonus is only significant when EI is small
    # Use a soft combination
    final_score = score + exploration_bonus
    
    # Ensure all values are finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
