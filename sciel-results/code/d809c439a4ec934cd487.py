import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for Expected Improvement (Minimization)
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Cost is in [1, 2], so dividing by cost gives a mild preference for lower cost
    # Use a power slightly less than 1 to soften the penalty
    
    # Compute a combined score that balances improvement and cost
    # We want to maximize (improvement / cost)
    # But also want some exploration when uncertainty is high
    
    # Strategy: use EI / cost as the primary score, but add a small exploration term
    # The exploration term is based on sigma / cost to explore uncertain regions cheaply
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Primary: EI per unit cost
    ei_per_cost = ei / cost_safe
    
    # Secondary: uncertainty per unit cost (encourages exploring uncertain areas cheaply)
    uncertainty_per_cost = sigma / cost_safe
    
    # Blend: mostly EI/cost, with a small exploration component
    # The exploration weight should be small to not over-explore
    # A fixed small weight works well; adapt it slightly based on the scale of EI
    ei_mean = np.mean(ei)
    if ei_mean > 1e-12:
        explore_weight = min(0.1, 0.5 * np.mean(sigma) / ei_mean)
    else:
        explore_weight = 0.1
    
    scores = ei_per_cost + explore_weight * uncertainty_per_cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
