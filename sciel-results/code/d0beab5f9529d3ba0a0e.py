import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Estimate N based on time, but start with a solid guess
    # N=10^7 is very fast with NumPy. Let's try 20^7? No, 2*10^7.
    N = 2 * 10**7
    
    # Check if we have enough time. If not, reduce N.
    # Rough estimate: Sieve is O(N log N). 
    # For N=10^7, it's ~1-2 seconds.
    # For N=5*10^7, it's ~5-10 seconds.
    # Let's stick to 10^7 to be safe and fast, allowing for multiple runs if we wanted, 
    # but we only need one good result.
    
    # Actually, let's go slightly higher to find rarer collisions.
    # 5 * 10^7 is 400MB for sigma. Might be tight.
    # Let's use 10^7.
    
    N = 10**7
    
    # Initialize sigma array
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # for i in 1..N: sigma[j] += i for j multiples of i
    # Vectorized:
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Check time
    if time.time() > deadline - 10:
        # Fallback to a known good result if sieve took too long (unlikely)
        return {'M': 333312, 'k': [336, 372, 434]}

    # Compute M = k * sigma(k)
    k_vals = np.arange(1, N + 1, dtype=np.int64)
    M_vals = k_vals * sigma[1:]  # sigma[0] is unused/0
    
    # Filter M <= 10^18
    limit = 10**18
    mask = M_vals <= limit
    k_valid = k_vals[mask]
    M_valid = M_vals[mask]
    
    if len(M_valid) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Find the most frequent M
    # np.unique is efficient for this size
    unique_Ms, counts = np.unique(M_valid, return_counts=True)
    
    max_count = np.max(counts)
    
    # If max_count is 1, we have no collisions? Unlikely for N=10^7.
    # Find all M with this max count (if multiple, pick the first one or largest? 
    # The score is the number of solutions, so any M with max count is equally good.
    # We just need to return the k's for one such M.
    
    max_M = unique_Ms[np.argmax(counts)]
    
    # Get k values for this M
    # We need to map back from M_valid to k_valid
    # Since M_valid and k_valid are aligned, we can use boolean indexing again
    # Or use the inverse mapping from np.unique if we stored it, but we didn't.
    # Re-filtering is O(N), which is fast.
    
    k_solutions = k_valid[M_valid == max_M]
    
    # Convert to Python int/list for return
    M_int = int(max_M)
    k_list = [int(k) for k in k_solutions]
    
    # Ensure distinct (they are by construction as k_vals are distinct)
    # Ensure k_i <= isqrt(M). 
    # As derived, k <= sigma(k) implies k^2 <= k*sigma(k) = M, so k <= sqrt(M).
    # This holds for all k >= 1.
    
    return {'M': M_int, 'k': k_list}
