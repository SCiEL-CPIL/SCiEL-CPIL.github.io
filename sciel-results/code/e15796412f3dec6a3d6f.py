"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Global variable to track the number of acquisition calls
# This allows us to implement an adaptive beta that increases over time
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    _call_count += 1
    
    # Calculate adaptive beta
    # Using UCB1-style adaptive beta: sqrt(2 * log(1 + t))
    # This starts low (exploitation) and increases (exploration)
    beta = np.sqrt(2.0 * np.log(1.0 + _call_count))
    
    # Ensure sigma is non-negative to avoid numerical issues
    sigma = np.maximum(sigma, 0.0)
    
    # Calculate the cost-aware LCB score
    # For minimization, we want to find points with low mu and high uncertainty (sigma)
    # The term (f_best - mu) represents the potential improvement (exploitation)
    # The term (beta * sigma) represents the uncertainty bonus (exploration)
    # We divide by cost to prefer cheaper points
    
    # Numerator: Potential improvement + Exploration bonus
    numerator = f_best - mu + beta * sigma
    
    # Denominator: Cost (with epsilon for stability)
    denominator = cost + 1e-9
    
    # Score
    scores = numerator / denominator
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
