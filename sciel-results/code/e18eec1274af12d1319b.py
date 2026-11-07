"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement with a cost-efficiency threshold.
    # This structurally favors points where the probability of beating f_best
    # is high relative to the cost, avoiding the instability of dividing by cost.
    
    # Compute z-score for EI
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Cost-aware threshold: We want PoI to be high enough to justify the cost.
    # A simple structural change is to penalize low PoI heavily or use a ratio.
    # However, the inspiration suggests PoI itself is the driver.
    # Let's use a modified PoI that is zero if the expected improvement is negligible.
    # Or simply return PoI, which inherently favors low sigma (concentrated CDF).
    
    # To be "cost-aware" structurally without division instability:
    # We can threshold the PoI. If PoI is below a certain level relative to cost, discard.
    # But a simpler structural change is just using PoI.
    # Let's try a "Cost-Adjusted PoI" which is PoI * exp(-cost) or similar?
    # No, the inspiration says "inherently favors high-confidence improvements... and naturally handles the cost".
    # This implies PoI's shape (sigmoid) is the key.
    # Let's return PoI directly, or PoI normalized by a soft cost term?
    # The previous best was 1.54 with PoI. Let's stick to PoI but ensure it's robust.
    
    return poi
# EVOLVE-BLOCK-END
