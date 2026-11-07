"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using proper Pareto-frontier selection.
    
    Computes non-dominated sort over (EI, uncertainty, inverse-cost) and
    assigns higher scores to candidates on the efficient frontier, balancing
    exploitation, exploration, and cost without arbitrary weight tuning.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware normalization
    cost_safe = np.maximum(cost, 1e-8)
    
    # Three objectives (all higher-is-better for Pareto):
    # 1. Expected improvement
    obj1 = ei
    # 2. Uncertainty (exploration value)
    obj2 = sigma_safe
    # 3. Inverse cost (efficiency)
    obj3 = 1.0 / cost_safe
    
    n = len(mu)
    
    # Check time before expensive computation
    if time.time() > deadline - 15.0:
        scores = ei / np.sqrt(cost_safe)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Non-dominated sort using vectorized dominance counting
    # A point i dominates j if obj_i[k] >= obj_j[k] for all k and > for at least one k
    # We count how many points dominate each point
    
    # For efficiency with n=2000, use a chunked approach
    # Dominance count: for each point, count how many others dominate it
    
    # Vectorized dominance: 
    # point i dominates j if:
    #   obj1[i] >= obj1[j] AND obj2[i] >= obj2[j] AND obj3[i] >= obj3[j]
    #   AND at least one is strictly greater
    
    # Use broadcasting in chunks to avoid memory issues
    # n=2000, so 2000x2000 = 4M booleans per objective, manageable
    
    # Compute dominance matrix in chunks
    chunk_size = 500
    dominance_count = np.zeros(n, dtype=np.int32)
    
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        # For each i in [i_start, i_end), count j that dominate i
        # j dominates i if obj1[j] >= obj1[i] AND obj2[j] >= obj2[i] AND obj3[j] >= obj3[i]
        # AND (obj1[j] > obj1[i] OR obj2[j] > obj2[i] OR obj3[j] > obj3[i])
        
        # Extract chunk
        o1_chunk = obj1[i_start:i_end, np.newaxis]  # (chunk, 1)
        o2_chunk = obj2[i_start:i_end, np.newaxis]  # (chunk, 1)
        o3_chunk = obj3[i_start:i_end, np.newaxis]  # (chunk, 1)
        
        # Compare with all points
        ge1 = (obj1[np.newaxis, :] >= o1_chunk)  # (chunk, n)
        ge2 = (obj2[np.newaxis, :] >= o2_chunk)
        ge3 = (obj3[np.newaxis, :] >= o3_chunk)
        
        # At least one strictly greater
        gt1 = (obj1[np.newaxis, :] > o1_chunk)
        gt2 = (obj2[np.newaxis, :] > o2_chunk)
        gt3 = (obj3[np.newaxis, :] > o3_chunk)
        
        dominates = ge1 & ge2 & ge3 & (gt1 | gt2 | gt3)
        
        # Count dominators for each point in chunk
        # dominates has shape (chunk, n), where dominates[c, j] means j dominates point (i_start + c)
        # We want to count how many j dominate each i in the chunk
        # So sum along axis=1 gives count for each i in chunk
        dominance_count[i_start:i_end] = np.sum(dominates, axis=1)
        
        # Time check
        if time.time() > deadline - 10.0:
            break
    
    # Points with 0 dominators are on the Pareto front
    # Use a smooth penalty: points with more dominators get lower scores
    
    # Base score: cost-adjusted EI
    base_score = ei / np.sqrt(cost_safe)
    
    # Pareto front boost: points on the front (0 dominators) get a boost
    # Use a smooth decay based on dominance count
    # Normalize dominance count
    max_dom = np.max(dominance_count) if np.max(dominance_count) > 0 else 1
    dom_norm = dominance_count / max_dom
    
    # Exponential decay: front points get full score, dominated points get penalized
    # Use a gentle penalty to not completely discard useful candidates
    pareto_weight = np.exp(-2.0 * dom_norm)
    
    scores = base_score * pareto_weight
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() > deadline - 5.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
