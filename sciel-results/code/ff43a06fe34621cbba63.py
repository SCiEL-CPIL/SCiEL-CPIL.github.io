import numpy as np

def construct():
    # We search in a block near 10^12
    # Let's use a block size of 2 * 10^6 to have a good chance
    B = 2 * 10**6
    S = 10**12 - B
    
    # 1. Generate primes up to sqrt(S+B)
    limit = int(np.sqrt(S + B)) + 1
    # Simple sieve for primes
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # 2. Initialize phi and rem arrays
    # phi[n] will store the totient
    # rem[n] will store the remaining part of n after removing small prime factors
    # We use int64 to handle 10^12
    phi = np.arange(S, S + B, dtype=np.int64)
    rem = np.arange(S, S + B, dtype=np.int64)
    
    # 3. Sieve
    for p in primes:
        # Find the first multiple of p in [S, S+B)
        # k = ceil(S / p) * p
        k = (S + p - 1) // p * p
        if k >= S + B:
            continue
            
        # Indices in the array corresponding to multiples of p
        # local_idx = (k - S) + i*p
        # We can use slicing: start_idx = k - S, step = p
        start_idx = k - S
        indices = np.arange(start_idx, B, p)
        
        # Update phi
        # phi[indices] -= phi[indices] // p
        # This is vectorized
        
        # Update rem
        # We need to divide rem by p as long as it's divisible
        # This is the tricky part to vectorize fully without loops
        # But for large p, it's only 1 division.
        # For small p, it's more.
        # We can do a while loop on the array
        vals = rem[indices].copy()
        # Check if any are divisible
        # We do this iteratively
        # Optimization: Only do this if p is small?
        # For p > 100, max power is 2 (since 100^3 = 10^6 < 10^12, wait. 10^6^2 = 10^12. So p^2 can be <= 10^12. p^3 > 10^12 for p > 1000).
        # So for p > 1000, at most 2 divisions.
        # For p > 100, at most 4 divisions.
        # For p > 10, at most 12 divisions.
        # Most primes are large.
        
        # Let's just use a while loop, it's fast enough for 10^6 elements?
        # The number of elements touched is B/p.
        # Sum of B/p is small.
        # The overhead of the while loop is per prime.
        
        while True:
            mask = vals % p == 0
            if not np.any(mask):
                break
            vals[mask] //= p
        rem[indices] = vals
        
        phi[indices] -= phi[indices] // p
        
    # 4. Handle remaining prime factors
    # If rem[i] > 1, it is a prime factor q
    mask = rem > 1
    if np.any(mask):
        indices = np.where(mask)[0]
        # q = rem[indices]
        # phi[indices] -= phi[indices] // rem[indices]
        phi[indices] -= phi[indices] // rem[indices]
        
    # 5. Find max run of distinct values
    best_len = 0
    best_start = S
    seen = set()
    curr_start = S
    
    # Convert to list for faster access? Or just iterate numpy array
    # Numpy array iteration is slow in Python.
    # Let's convert to a Python list of ints? 2M ints is ~16MB. OK.
    # Or just iterate.
    
    # Actually, we can just iterate.
    for i in range(B):
        val = phi[i]
        if val in seen:
            # Reset
            seen = set()
            curr_start = S + i
        else:
            seen.add(val)
            curr_len = (S + i) - curr_start + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = curr_start
            if best_len == 200: # Max allowed
                break
                
    # If we didn't find 200, we might want to try another block?
    # Given the constraints, a run of length > 20 is likely found.
    # Let's return the best found.
    
    return {'start': int(best_start), 'length': int(best_len)}
