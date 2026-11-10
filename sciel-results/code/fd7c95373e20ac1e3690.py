import time
import math
import numpy as np
from collections import defaultdict

def construct():
    deadline = time.time() + 135  # 135s deadline, 15s safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = len(best_k)
    
    # Try to find better solutions by scanning larger ranges
    # Use progressively larger N values
    # Memory constraint: sigma (int32) + M (int64) = 12 bytes per element.
    # 10^8 elements = 1.2 GB. Might be tight. Let's try 5*10^7 first.
    N_values = [10**6, 2*10**6, 5*10**6, 10**7, 2*10**7, 5*10**7]
    
    for N in N_values:
        if time.time() > deadline - 10:
            break
            
        try:
            # Compute sigma using NumPy
            # Use int32 to save memory. Max sigma for 5*10^7 is < 2*10^9?
            # sigma(n) <= n * (1 + 1/2 + ... ) ~ n log log n.
            # For n=5e7, log log n ~ 3. sigma ~ 1.5e8. Fits in int32.
            sigma = np.zeros(N + 1, dtype=np.int32)
            for i in range(1, N + 1):
                sigma[i::i] += i
            
            # Compute M = k * sigma(k)
            # k is 1..N
            ks = np.arange(1, N + 1, dtype=np.int64)
            # Ensure sigma is int64 for multiplication to avoid overflow in intermediate?
            # int32 * int64 -> int64.
            M = ks * sigma
            
            # Filter M <= 10^18 (should always be true for N=5e7)
            # M max ~ 5e7 * 1.5e8 = 7.5e15 < 1e18
            
            # Find duplicates
            # np.unique is O(N log N) but fast in C
            unique_M, inverse_indices = np.unique(M, return_inverse=True)
            
            # Count occurrences
            counts = np.bincount(inverse_indices)
            
            # Find max count
            max_count = np.max(counts)
            
            if max_count > best_count:
                # Find indices where count is max
                # We need the M value and the k values
                max_indices = np.where(counts == max_count)[0]
                # Pick the first one
                idx = max_indices[0]
                best_M_val = unique_M[idx]
                
                # Find all k such that M[k] == best_M_val
                # We can use the inverse_indices or just search M
                # Since we have inverse_indices, we can find where inverse_indices == idx
                # But inverse_indices is length N.
                # Better: np.where(M == best_M_val)[0]
                # This is O(N).
                mask = (M == best_M_val)
                k_vals = ks[mask]
                
                best_M = int(best_M_val)
                best_k = [int(k) for k in k_vals]
                best_count = max_count
                
                # Early exit if we find a very good solution
                if best_count >= 10:
                    break
                    
        except Exception as e:
            # If memory error or other, break
            break
            
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    # Verify the solution
    for k in best_k:
        if k > math.isqrt(best_M):
            # Fallback
            return {'M': 333312, 'k': [336, 372, 434]}
            
    return {'M': best_M, 'k': best_k}
