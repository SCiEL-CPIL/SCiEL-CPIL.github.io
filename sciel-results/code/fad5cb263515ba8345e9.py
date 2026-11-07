"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Hard cost gate: prioritize cheap candidates with high EI,
    # heavily damp expensive candidates to exhaust cheap region first.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Running average cost as threshold
    avg_cost = np.mean(cost)
    
    # Hard gate: full EI for cheap candidates, strong damping for expensive
    # Use a smooth step to avoid discontinuities but maintain strong preference
    # Candidates below 0.8*avg_cost get full EI
    # Candidates above 1.2*avg_cost get heavily damped EI
    threshold_low = 0.8 * avg_cost
    threshold_high = 1.2 * avg_cost
    
    # Create gating factor: 1.0 for cheap, 0.1 for expensive
    gate = np.where(
        cost <= threshold_low,
        1.0,
        np.where(
            cost >= threshold_high,
            0.1,
            # Linear interpolation between thresholds
            1.0 - 0.9 * (cost - threshold_low) / (threshold_high - threshold_low + 1e-9)
        )
    )
    
    return ei * gate
# EVOLVE-BLOCK-END
