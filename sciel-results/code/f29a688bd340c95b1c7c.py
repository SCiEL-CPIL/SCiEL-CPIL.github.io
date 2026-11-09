# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
        # Use Expected Improvement (EI) for minimization with a jitter.
        # The jitter makes the acquisition more conservative, requiring
        # a significant expected improvement to select a point.
        
        # Clip sigma to avoid division by zero and numerical instability
        sigma = np.maximum(sigma, 1e-8)
        
        # Add a small jitter to f_best to encourage more significant improvements
        # This prevents the algorithm from getting stuck in local optima by
        # requiring a stronger signal to move.
        jitter = 0.01 * abs(f_best)
        f_best_eff = f_best - jitter
        
        # Calculate the normalized distance from the best found value
        # z = (f_best_eff - mu) / sigma
        z = (f_best_eff - mu) / sigma
        
        # Compute CDF and PDF
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        
        # Calculate EI scores
        scores = (f_best_eff - mu) * cdf_z + sigma * pdf_z
        
        # Ensure finite scores
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        return scores
    # EVOLVE-BLOCK-END
