"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending normalized EI and PI
    with a dynamic uncertainty floor for robust exploration-exploitation balance.
    
    Parameters:
    -----------
    mu : array_like, shape (n,)
        Mean of the GP surrogate at candidate points.
    sigma : array_like, shape (n,)
        Standard deviation of the GP surrogate at candidate points.
    f_best : float
        Best observed value so far (lowest value, since we minimize).
    cost : array_like, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (PI)
    # PI = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Dynamic uncertainty floor: blend EI and PI based on how much we've explored
    # When f_best is good (low), we should exploit more (weight EI higher)
    # When f_best is poor (high relative to typical values), explore more (weight PI higher)
    # We use a simple heuristic: as f_best decreases, increase exploitation weight
    
    # Normalize EI and PI to [0, 1] range for blending
    # EI can be negative if mu > f_best, so clip to non-negative
    ei_clipped = np.maximum(ei, 0.0)
    
    # Scale EI by typical improvement magnitude
    # Use the median of positive EI values as a scale, or fall back to a small value
    positive_ei = ei_clipped[ei_clipped > 0]
    if len(positive_ei) > 0:
        ei_scale = np.median(positive_ei)
        if ei_scale < 1e-10:
            ei_scale = 1e-10
    else:
        ei_scale = 1e-10
    
    ei_normalized = ei_clipped / ei_scale
    
    # PI is already in [0, 1]
    
    # Dynamic blending weight: more exploration when we're early (f_best is poor)
    # Use a sigmoid-like function based on f_best
    # For hartmann 3d, the minimum is around -3.86
    # If f_best is close to the minimum, exploit; if far, explore
    # Simple heuristic: weight_ei = 1 / (1 + exp(-f_best))
    # When f_best is very negative (good), weight_ei approaches 1 (exploit)
    # When f_best is positive (bad), weight_ei approaches 0.5
    # Let's use a different approach: based on the number of evaluations
    # But we don't have that info directly. Use f_best as a proxy.
    
    # Alternative: use a fixed blend that works well in practice
    # From the knowledge: blending normalized EI and PI with dynamic uncertainty floor
    # worked well (0.7286 -> 2.249)
    
    # Let's use a dynamic weight based on the ratio of sigma to (f_best - mu)
    # High uncertainty relative to improvement -> more exploration
    uncertainty_ratio = sigma / np.maximum(np.abs(f_best - mu), 1e-10)
    
    # Blend weight: higher when uncertainty is high
    # weight_ei = 1 / (1 + uncertainty_ratio * 0.5)
    # This gives more weight to EI when uncertainty is low (exploit)
    # and more weight to PI when uncertainty is high (explore)
    weight_ei = 1.0 / (1.0 + 0.5 * uncertainty_ratio)
    weight_pi = 1.0 - weight_ei
    
    # Blended acquisition before cost normalization
    blended = weight_ei * ei_normalized + weight_pi * pi
    
    # Cost-aware normalization: divide by cost
    scores = blended / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
