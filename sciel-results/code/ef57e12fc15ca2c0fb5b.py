import math
import numpy as np

def construct():
    # We'll search for M <= 10^18 with many k <= sqrt(M) such that k*sigma(k) = M
    # Strategy: enumerate k up to a limit, compute k*sigma(k), count multiplicities
    # Use a NumPy-based sieve to compute sigma(k) efficiently
    
    # Set a reasonable limit for k enumeration
    # k <= sqrt(M) and M <= 10^18, so k <= 10^9
    # But we can't enumerate to 10^9, so we'll enumerate to a practical limit
    # and hope to find good M values
    
    # Let's try enumerating k up to 10 * 10^6
    N = 10000000
    
    # Sieve for sigma function
    # sigma(n) = sum of divisors
    # We can compute it using a modified sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Now compute k * sigma(k) for each k and count
    k_arr = np.arange(1, N + 1, dtype=np.int64)
    M_vals = k_arr * sigma[1:]
    
    # Find M with most solutions
    unique_M, counts = np.unique(M_vals, return_counts=True)
    
    if len(unique_M) == 0:
        return {'M': 1, 'k': [1]}
        
    max_count = np.max(counts)
    idx = np.argmax(counts)
    M_star = int(unique_M[idx])
    
    # Filter k <= sqrt(m)
    sqrt_M = math.isqrt(M_star)
    mask = (M_vals == M_star) & (k_arr <= sqrt_M)
    k_list = k_arr[mask].tolist()
    
    # Ensure types are correct
    M_star = int(M_star)
    k_list = [int(k) for k in k_list]
    
    return {'M': M_star, 'k': k_list}
