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
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal.
    ei = (f_best - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Compute Probability of Improvement (POI)
    poi = stats.norm.cdf(z)
    
    # Cost-aware scaling:
    # Use a quadratic-like penalty to strongly favor cheap points.
    # score_ceil = EI / (cost * (1 + cost))
    # This is the structural idea from Inspiration 1.
    
    # Refinement: Blend in POI to handle edge cases.
    # If EI is very small (e.g., mu >> f_best), EI/cost will be near 0.
    # But if POI is high (which implies mu < f_best, so EI should be positive),
    # we might miss some points. However, if mu < f_best, EI is always positive.
    # So POI is only useful when mu > f_best? No, if mu > f_best, POI < 0.5.
    # Actually, POI is the probability that f(x) < f_best.
    # If mu < f_best, POI > 0.5 and EI > 0.
    # If mu > f_best, POI < 0.5 and EI > 0 (due to the sigma*phi(z) term).
    # So EI is always positive.
    
    # The main issue with pure EI/cost is that it might be too aggressive in exploration
    # if sigma is large. But CEI with quadratic cost penalty should handle this.
    
    # Let's stick to the CEI structure but ensure numerical stability.
    scores = ei / (cost * (1.0 + cost))
    
    # Optional: Add a small POI term to boost exploration in uncertain regions?
    # The current program (POI/sqrt(cost)) scored 1.25.
    # Inspiration 1 (EI/(cost*(1+cost))) scored 1.44.
    # Let's try a hybrid: score = EI / (cost * (1 + cost)) + lambda * POI / cost
    # But we want to keep it simple and robust. The CEI term is likely dominant.
    
    # Let's refine the cost penalty slightly.
    # Maybe cost^1.5 is better than cost*(1+cost)?
    # cost*(1+cost) = cost + cost^2.
    # For cost in [1, 2] (since cost = 1 + mean(x_scaled)), cost^2 is 1-4.
    # This strongly penalizes expensive points.
    
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
