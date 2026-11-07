"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    divided by a cost-aware factor.
    
    The cost of evaluating a point is 1 + mean(x) where x is in [0,1]^3.
    Since we don't have x directly, we use the provided cost array.
    
    We use a soft cost-aware weighting that balances exploration (high EI)
    and exploitation (low cost). The acquisition score is:
    
    score = EI / (cost^alpha)
    
    where alpha is tuned to be mild (0.5) to avoid overly penalizing
    expensive points while still preferring cheaper evaluations.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (clip small negative values from numerical error)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware weighting: divide by cost^alpha
    # cost is in [1, 2] for x in [0,1]^3
    # Use alpha=0.5 for a mild penalty
    alpha = 0.5
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = np.power(cost_safe, alpha)
    
    # Acquisition score
    score = ei / cost_factor
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
