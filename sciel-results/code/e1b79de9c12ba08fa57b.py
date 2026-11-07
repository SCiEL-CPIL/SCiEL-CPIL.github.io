import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Probability of Improvement (POI) and Expected Improvement (EI),
    both normalized by cost, with an exploration bonus for high-uncertainty regions.
    """
    # Ensure sigma is non-negative and handle very small values
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Expected Improvement
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * poi + sigma * pdf_z
    
    # Cost-aware scoring: divide by cost to penalize expensive points
    # Cost is 1 + mean(x_scaled), so ranges from 1 to 2
    
    # Combine POI and EI with weighting
    # POI is more robust and favors exploration
    # EI is more aggressive and favors exploitation
    # Use a blend that emphasizes POI for robustness
    weighted_score = 0.7 * poi + 0.3 * (ei / (np.abs(f_best) + 1e-6))
    
    # Add exploration bonus for high-uncertainty regions
    # This helps escape local minima and explore the Rosenbrock valley
    exploration_bonus = 0.05 * sigma / (np.abs(f_best) + 1e-6)
    
    # Combine and normalize by cost
    score = (weighted_score + exploration_bonus) / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
