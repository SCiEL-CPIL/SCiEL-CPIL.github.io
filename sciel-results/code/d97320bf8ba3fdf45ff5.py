"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats as stats

# Deadline guard: The evaluator kills after 330s.
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Improvement (CEI) with Probability of Improvement (POI) blending.
    
    This function combines the structural idea of CEI (maximizing expected improvement
    per unit cost, specifically using a quadratic cost penalty to strongly favor cheap
    points) with the refinement of blending in POI to handle cases where EI is zero
    but there is still a high probability of improvement (e.g., when mu is close to 
    f_best but sigma is large enough that there's a chance of going lower).
    
    The core score is EI / (cost * (1 + cost)).
    To ensure we don't ignore points with high POI but low EI (which can happen if 
    mu is slightly above f_best but sigma is large), we add a small weighted POI term.
    However, the primary driver should remain the cost-aware EI.
    
    Parameters:
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        The best observed function value so far.
    cost : np.ndarray, shape (N,)
        Cost of evaluating each candidate.
    
    Returns:
    scores : np.ndarray, shape (N,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute Probability of Improvement (POI)
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    poi = stats.norm.cdf(z)
    
    # Cost-aware scaling:
    # CD-PI strategy: Maximize the frequency of successful updates per unit cost.
    # Score = POI / sqrt(cost)
    # This favors points with a high probability of improving the best,
    # with a softer penalty for cost compared to quadratic penalties.
    scores = poi / np.sqrt(cost)
    
    # Let's check the deadline.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
         # Fallback: return current scores
         scores = np.where(np.isfinite(scores), scores, 0.0)
         return scores
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
