import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement and Lower Confidence Bound,
    weighted by cost. Uses objective-unit normalization to avoid fragile min-max scaling.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Lower Confidence Bound (LCB) for minimization
    # Use a moderate exploration parameter
    xi = 0.01
    lcb = mu - xi * sigma
    
    # Normalize EI using a soft scaling based on f_best magnitude
    # This avoids the fragility of min-max normalization when values are close
    ei_scale = np.maximum(f_best, 1e-6)
    ei_norm = ei / ei_scale
    
    # LCB: lower is better, so we want to maximize (f_best - lcb)
    # This gives the expected improvement under LCB
    lcb_improvement = np.maximum(f_best - lcb, 0)
    lcb_norm = lcb_improvement / ei_scale
    
    # Combine: 70% EI, 30% LCB
    # Both are already in objective units, so this is a principled combination
    combined = 0.7 * ei_norm + 0.3 * lcb_norm
    
    # Cost-aware: divide by cost to balance improvement vs. evaluation cost
    scores = combined / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
