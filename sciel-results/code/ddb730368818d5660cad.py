import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    # We will search up to this limit for k
    # Based on previous runs, 20M was safe. With vectorized grouping, we can go higher.
    # Let's try 50M. If it's too slow, we'll fall back or the deadline guard will catch it.
    N_MAX = 50000000

    # Precompute sigma for all k using a divisor-sum sieve with NumPy
    # Use int64 to be safe with sigma values
    sigma_arr = np.zeros(N_MAX + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This loop is O(N log N) but vectorized inner loop
    for d in range(1, N_MAX + 1):
        if time.time() > deadline - 30:
            break
        # Add d to all multiples of d
        # d::d creates a view, += d modifies in place
        sigma_arr[d::d] += d

    # Verify with a quick spot check
    assert int(sigma_arr[1]) == 1
    assert int(sigma_arr[2]) == 3
    assert int(sigma_arr[6]) == 12

    # Compute M = k * sigma(k)
    # k is 1..N_MAX
    k_arr = np.arange(1, N_MAX + 1, dtype=np.int64)
    
    # Check for overflow or M > 10^18
    # We need to be careful with multiplication overflow in int64.
    # Max k ~ 5e7. Max sigma(k) ~ 5e7 * 3 = 1.5e8. Product ~ 7.5e15 < 1e18.
    # So int64 is safe for the product.
    M_arr = k_arr * sigma_arr[1:] # sigma_arr[1:] corresponds to k=1..N_MAX

    # Filter valid M: M <= 10^18
    # Note: k <= isqrt(M) is always true for k>=1 as established.
    valid_mask = M_arr <= 10**18
    valid_M = M_arr[valid_mask]
    valid_k = k_arr[valid_mask]
    
    if len(valid_M) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Find the M with the most occurrences
    # np.unique with return_counts and return_inverse
    # unique_M, inverse_indices, counts = np.unique(valid_M, return_inverse=True, return_counts=True)
    # This is efficient.
    
    unique_M, inverse_indices, counts = np.unique(valid_M, return_inverse=True, return_counts=True)
    
    # Find the index of the max count
    max_idx = np.argmax(counts)
    best_count = counts[max_idx]
    best_M_val = unique_M[max_idx]
    
    # If we didn't find any better than 4, we might still return the best found.
    # But we want to maximize.
    
    # Extract k values for best_M_val
    # We need the k's that mapped to best_M_val
    # inverse_indices tells us which unique_M index each valid_M element belongs to.
    # So valid_k[inverse_indices == max_idx] are the solutions.
    best_ks = valid_k[inverse_indices == max_idx]
    
    # Convert to list of Python ints
    best_ks_list = [int(x) for x in best_ks]
    
    # Final verification and type conversion
    # Re-verify sigma for the final list using exact computation
    def sigma_exact(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                p_power = 1
                while temp % p == 0:
                    temp //= p
                    exp += 1
                    p_power *= p
                result *= (p_power * p - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp * temp - 1) // (temp - 1)
        return result

    final_ks = []
    for k in best_ks_list:
        if isinstance(k, int) and 1 <= k <= math.isqrt(best_M_val):
            if k * sigma_exact(k) == best_M_val:
                final_ks.append(k)
    
    if not final_ks:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M_val), 'k': final_ks}
