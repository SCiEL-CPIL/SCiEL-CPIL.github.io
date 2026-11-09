import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    
    # Blend EI and PoI
    # PoI encourages exploration of regions with high uncertainty or potential improvement
    # EI focuses on expected gain
    # A common blend is to add a fraction of PoI to EI
    # Let's try a 50/50 blend or similar, scaled by cost
    # The "Knowledge" suggests PoI was key to a jump in score.
    # Let's try: Score = (EI + PoI) / Cost
    # This treats PoI as an additional unit of "value" similar to EI,
    # but EI is in objective units while PoI is probability.
    # To make them comparable, we might need to scale PoI.
    # However, if f_best - mu is large, EI is large.
    # If we just add them, EI might dominate.
    # Let's try a weighted sum: 1.0 * EI + 1.0 * PoI * sigma_scale?
    # No, let's keep it simple first.
    
    # Alternative: UCB (Upper Confidence Bound)
    # UCB = mu - kappa * sigma
    # But we need cost awareness.
    
    # Let's stick to the EI/PoI blend with linear cost normalization.
    # The previous program used cost**2. Let's try cost**1.
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Combine EI and PoI
    # Weights: EI is the primary driver. PoI helps escape local optima.
    # Let's try EI + 0.5 * PoI
    combined = ei + 0.5 * poi
    
    scores = combined / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
