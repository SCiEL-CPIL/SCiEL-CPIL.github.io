import math
import numpy as np

def construct():
    # Set a limit that balances search space and computation time
    # 10^7 is a good balance for Python + NumPy within 120s
    N = 10_000_000
    
    # 1. Compute sigma(n) for n in [1, N]
    sigma = np.zeros(N + 1, dtype=np.int64)
    # Vectorized sieve for sigma
    # This loop is the bottleneck for Python overhead, but vectorized ops are fast
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # 2. Compute M_k = k * sigma(k)
    ks = np.arange(1, N + 1, dtype=np.int64)
    vals = ks * sigma[1:N+1]
    
    # 3. Filter M <= 10^18
    max_M = 10**18
    mask = vals <= max_M
    vals_f = vals[mask]
    ks_f = ks[mask]
    
    if len(vals_f) == 0:
        return {'M': 1, 'k': [1]}
        
    # 4. Find M with maximum frequency
    # unique sorts the array, which is efficient in C
    unique_vals, counts, inverse = np.unique(vals_f, return_counts=True, return_inverse=True)
    
    # Find the index of the most frequent value
    max_idx = np.argmax(counts)
    best_M = unique_vals[max_idx]
    
    # 5. Retrieve all k for this M
    # inverse[j] == max_idx means vals_f[j] == best_M
    k_indices = np.where(inverse == max_idx)[0]
    best_ks = ks_f[k_indices]
    
    # 6. Filter k <= sqrt(M) (though theoretically all should pass)
    sqrt_M = math.isqrt(best_M)
    valid_ks = best_ks[best_ks <= sqrt_M]
    
    # 7. Convert to Python types
    best_M = int(best_M)
    valid_ks = [int(k) for k in valid_ks]
    
    if not valid_ks:
        return {'M': 1, 'k': [1]}
        
    return {'M': best_M, 'k': valid_ks}
