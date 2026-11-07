import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin

    # Determine N
    # Based on previous performance, 10^7 takes ~10-15s.
    # Let's try 50,000,000.
    N = 50_000_000
    
    # If we are very fast, we could go higher, but 50M is a good balance.
    # Memory: 
    # sigma (int32): 200 MB
    # k_arr (int32): 200 MB
    # M_arr (int64): 400 MB
    # Total: 800 MB. Safe.

    # Sieve for sigma
    sigma = np.zeros(N + 1, dtype=np.int32)
    
    # Check time periodically
    # The loop is the bottleneck.
    for i in range(1, N + 1):
        sigma[i::i] += i
        
        if i % 5_000_000 == 0:
            if time.time() - start_time > DEADLINE - 40:
                # If we are running out of time, we cannot finish the sieve.
                # We must return a fallback.
                # But breaking here leaves sigma incomplete.
                # So we should have chosen a smaller N.
                # Since we can't change N now, we return seed.
                return {'M': 333312, 'k': [336, 372, 434]}

    # Compute M
    k_arr = np.arange(1, N + 1, dtype=np.int32)
    # Ensure no overflow in multiplication
    M_arr = k_arr.astype(np.int64) * sigma[1:].astype(np.int64)
    
    if time.time() - start_time > DEADLINE - 20:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Filter M <= 10^18
    valid_mask = M_arr <= 10**18
    M_valid = M_arr[valid_mask]
    k_valid = k_arr[valid_mask]
    
    if len(M_valid) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Find collisions using np.unique
    # This sorts M_valid, which is O(N log N)
    unique_M, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
    
    if time.time() - start_time > DEADLINE - 10:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Find the M with max count
    max_idx = np.argmax(counts)
    best_count = counts[max_idx]
    
    if best_count < 3:
        # If we didn't find better than seed, return seed
        return {'M': 333312, 'k': [336, 372, 434]}

    best_M = unique_M[max_idx]
    
    # Get the k values for this M
    best_k_indices = np.where(inverse == max_idx)[0]
    best_k = k_valid[best_k_indices]
    
    # Convert to Python list
    best_k_list = [int(k) for k in best_k]
    
    # Verify constraint k <= sqrt(M)
    sqrt_M = math.isqrt(int(best_M))
    filtered_k = [k for k in best_k_list if k <= sqrt_M]
    
    if len(filtered_k) >= len(best_k_list):
         # All valid
         return {'M': int(best_M), 'k': filtered_k}
    else:
         # Some k were > sqrt(M), which shouldn't happen for k*sigma(k)=M, k>1
         # But if it does, filter them.
         # If filtering reduces count below 3, return seed.
         if len(filtered_k) >= 3:
             return {'M': int(best_M), 'k': filtered_k}
         else:
             return {'M': 333312, 'k': [336, 372, 434]}
