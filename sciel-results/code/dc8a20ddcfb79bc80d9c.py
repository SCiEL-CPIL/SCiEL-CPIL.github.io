"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to cache random numbers if needed, but we generate fresh each time
# We need to ensure we are fast. 2000 candidates, 30 evals.
# KG approximation:
# Q_i = min_{j != i} (Y_j - Y_i)
# KG_i = E[ max(0, Q_i) ]
# Y ~ N(mu, sigma^2)

def acquisition(mu, sigma, f_best):
    # Time guard
    start_time = time.time()
    deadline = 290  # Safety margin from 300s limit
    
    n = len(mu)
    
    # Normalize sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # We use a Monte Carlo estimate of the Knowledge Gradient.
    # Number of MC samples. 
    # 2000 candidates. 
    # If we do M samples, we have an M x N matrix of samples.
    # We need to find the second minimum for each column in the sample?
    # Actually, Q_i = min_{j != i} Y_j - Y_i.
    # If Y_i is the global min, Q_i = second_min - Y_i.
    # If Y_i is not the global min, Q_i = min - Y_i.
    # In both cases, Q_i = min_{j != i} Y_j - Y_i.
    
    # Efficient calculation:
    # For each sample vector Y (length N):
    # 1. Find indices of min and second min.
    # 2. For all i, min_{j!=i} Y_j is:
    #    - If Y_i == min_val, then second_min_val
    #    - Else min_val
    # 3. Q_i = min_{j!=i} Y_j - Y_i
    
    # We can vectorize this over M samples.
    # However, M * N memory might be large if M is large.
    # N=2000. M=100 -> 200,000 floats. Very small.
    # M=1000 -> 2,000,000 floats. 16MB. Fine.
    
    # Let's use M=1000 samples for accuracy.
    M = 1000
    
    # Generate samples: (M, N)
    # Y = mu + sigma * Z
    # Z is standard normal
    
    # To save time, we can generate Z in chunks or just all at once.
    # np.random.randn(M, N) is fast.
    
    # Check time before heavy allocation?
    if time.time() > deadline - 10:
        # Fallback to EI if we are running out of time
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    Z = np.random.randn(M, n)
    Y = mu[None, :] + sigma_safe[None, :] * Z
    
    # Find min and second min for each sample (row)
    # argsort is O(N log N). M * N log N.
    # 1000 * 2000 * 11 ~ 22M ops. Fast in numpy.
    
    # We can use argpartition to find the smallest 2 elements faster?
    # Or just argsort.
    
    # Let's try argsort.
    # indices = np.argsort(Y, axis=1)
    # min_idx = indices[:, 0]
    # second_min_idx = indices[:, 1]
    # min_val = Y[np.arange(M), min_idx]
    # second_min_val = Y[np.arange(M), second_min_idx]
    
    # This creates large intermediate arrays.
    # Alternative:
    # min_val = np.min(Y, axis=1)
    # # To find second min, we can mask the min?
    # # Or use argpartition.
    
    # argpartition is O(N).
    # part = np.argpartition(Y, 1, axis=1)
    # min_idx = part[:, 0]
    # second_min_idx = part[:, 1]
    # # Note: argpartition(1) gives the first 2 smallest, but not sorted.
    # # So min_val is Y[:, min_idx], second_min_val is Y[:, second_min_idx]
    # # But we need to ensure which one is actually the min.
    # # Actually, if we just want min_{j!=i} Y_j:
    # # For a given sample row:
    # # Let m1 be min, m2 be second min.
    # # If Y_i == m1, min_{j!=i} is m2.
    # # If Y_i != m1, min_{j!=i} is m1.
    
    # Let's use argpartition for speed.
    # part = np.argpartition(Y, 1, axis=1)
    # idx0 = part[:, 0]
    # idx1 = part[:, 1]
    # val0 = Y[np.arange(M), idx0]
    # val1 = Y[np.arange(M), idx1]
    
    # # Determine actual min and second min
    # # min_val = np.minimum(val0, val1)
    # # second_min_val = np.maximum(val0, val1)
    
    # # Now for each element Y_i:
    # # is_min = (Y == min_val[:, None])  # (M, N) boolean
    # # min_excl_i = np.where(is_min, second_min_val[:, None], min_val[:, None])
    # # Q = min_excl_i - Y
    # # KG = np.mean(np.maximum(Q, 0), axis=0)
    
    # This creates (M, N) boolean and (M, N) float arrays.
    # M=1000, N=2000. 2M elements. 16MB each. Total ~50MB. Acceptable.
    
    # Let's implement this.
    
    try:
        part = np.argpartition(Y, 1, axis=1)
        idx0 = part[:, 0]
        idx1 = part[:, 1]
        
        val0 = Y[np.arange(M), idx0]
        val1 = Y[np.arange(M), idx1]
        
        min_val = np.minimum(val0, val1)
        second_min_val = np.maximum(val0, val1)
        
        # Broadcasting: min_val is (M,), Y is (M, N)
        # We need to check if Y[i] is the min.
        # Due to floating point, exact equality might fail, but since we took values from Y, it should be fine.
        # However, if there are ties, it's ambiguous. Probability of tie is 0.
        
        is_min = (Y == min_val[:, None])
        
        # min_excl_i: if Y is min, use second_min, else use min
        min_excl_i = np.where(is_min, second_min_val[:, None], min_val[:, None])
        
        Q = min_excl_i - Y
        
        # Knowledge Gradient is E[max(0, Q)]
        # We want higher score to be better.
        # KG estimates the expected reduction in the minimum value.
        # So we return the KG value.
        
        kg_scores = np.mean(np.maximum(Q, 0), axis=0)
        
        # Check time
        if time.time() > deadline - 5:
             # Fallback if calculation took too long (unlikely for 1000x2000)
             z = (f_best - mu) / sigma_safe
             return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        return kg_scores

    except Exception:
        # Fallback to EI on any error
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
# EVOLVE-BLOCK-END
