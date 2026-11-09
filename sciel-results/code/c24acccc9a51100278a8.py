"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blends Expected Improvement (exploitation) with
    Lower Confidence Bound (exploration) using an adaptive weight that
    increases exploration pressure when the best improvement is small
    (suggesting we may be stuck in a local region).
    
    Also applies a mild entropy bonus to encourage diverse sampling.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Lower Confidence Bound (higher score = better, so we negate the penalty)
    # LCB for minimization: mu - kappa * sigma
    # As a "score" where higher is better: -mu + kappa * sigma
    # But we want to blend with EI which is already a positive score for improvement.
    # Let's use a different form: the probability-weighted improvement.
    
    # Alternative: Use a weighted combination where we add a bonus for uncertainty
    # that is scaled by how much improvement is currently expected.
    
    # Adaptive kappa: larger when EI is small (stagnation signal)
    # If EI is near zero everywhere, we need more exploration.
    max_ei = np.max(ei)
    if max_ei < 1e-6:
        # Very little improvement expected; boost exploration significantly
        kappa = 3.0
    elif max_ei < 0.1:
        kappa = 2.0
    elif max_ei < 1.0:
        kappa = 1.5
    else:
        kappa = 1.0
    
    # LCB score: higher is better
    lcb_score = -mu + kappa * sigma_safe
    
    # Normalize EI to be comparable to LCB score
    # EI is in the range [0, f_best - min(mu)], while LCB can be negative.
    # Let's blend them by adding a scaled LCB term to EI.
    
    # The EI already accounts for both mu and sigma. Adding LCB explicitly
    # might double-count. Instead, let's use a different approach:
    # Use a "probability of improvement" weighted term plus a pure uncertainty term.
    
    # PI (Probability of Improvement): P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z_ei)
    
    # Blend: w * EI + (1-w) * sigma_safe * pi
    # When w=1, it's pure EI. When w=0, it's uncertainty-weighted PI.
    # The uncertainty-weighted PI encourages exploration in high-sigma regions
    # that still have a reasonable chance of beating f_best.
    
    # Adaptive weight: start with more EI, shift to exploration if EI is flat
    if max_ei < 1e-6:
        w = 0.3
    elif max_ei < 0.01:
        w = 0.5
    elif max_ei < 0.1:
        w = 0.7
    else:
        w = 0.9
    
    score = w * ei + (1 - w) * sigma_safe * pi
    
    # Add a small bonus for regions with high sigma to encourage diverse sampling
    # This helps escape local optima by giving a slight edge to uncertain regions
    sigma_bonus = 0.1 * sigma_safe
    
    return score + sigma_bonus
# EVOLVE-BLOCK-END
