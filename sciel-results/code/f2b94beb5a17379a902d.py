import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # Ensure sigma is not zero to avoid division by zero
        sigma_safe = np.maximum(sigma, 1e-6)
        
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        
        cdf = norm.cdf(z_clipped)
        pdf = norm.pdf(z_clipped)
        
        # Standard EI
        ei = (f_best - mu) * cdf + sigma_safe * pdf
        ei = np.maximum(ei, 0.0)
        
        # Small exploration bonus to break ties in flat regions
        # Using a very small coefficient to not disturb the EI structure
        exploration_bonus = 0.05 * sigma_safe
        
        return ei + exploration_bonus
