import math
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # We want to find M with many solutions k such that k * sigma(k) = M
    # k <= sqrt(M) is automatically satisfied if M = k * sigma(k) and sigma(k) >= k+1 (for k>1)
    # For k=1, M=1, k<=1.
    
    # Linear Sieve for sigma
    # We estimate N. 
    # Time for N=10^7 in Python linear sieve is approx 5-10s.
    # Let's try N = 10^7.
    N = 10_000_000
    
    # Check if we have enough time
    # If not, reduce N.
    # But we don't know time per iter exactly.
    # Let's just run N=10^7.
    
    sigma = [1] * (N + 1)
    p_pow = [1] * (N + 1)
    p_sigma = [1] * (N + 1)
    spf = [0] * (N + 1) # Smallest prime factor
    primes = []
    
    # sigma[1] = 1, p_pow[1]=1, p_sigma[1]=1 are already set
    
    best_M = 0
    best_k_list = []
    max_count = 0
    
    # We need a dictionary to store solutions for each M
    # But M can be up to 10^18.
    # We can just update best_M on the fly if we find a collision?
    # No, we need to know the count for each M.
    # Using a dict: M -> count (or list)
    # Storing lists is memory heavy if many M's.
    # But most M's will have count 1.
    # We only care about M's with count > 1.
    # We can store count in a dict, and only store the list if count > 1?
    # Or store list for all, but that's 10^7 lists. Too much memory.
    
    # Strategy:
    # 1. Pass 1: Compute sigma[i] for all i. Store in array.
    # 2. Pass 2: Iterate i, compute M = i * sigma[i].
    #    If M <= 10^18:
    #       if M in counts:
    #           counts[M] += 1
    #           if counts[M] > 2: # Only track details for likely winners?
    #               # We need the list of k's.
    #               # We can rebuild the list for the best M at the end.
    #       else:
    #           counts[M] = 1
    #    Track max_count and best_M.
    # 3. After finding best_M, iterate i again to collect all k such that i * sigma[i] == best_M.
    
    # This requires 2 passes.
    # Pass 1: Sieve.
    # Pass 2: Count frequencies.
    # Pass 3: Collect k's for best_M.
    
    # Let's implement Pass 1.
    
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = i + 1
            p_pow[i] = i
            p_sigma[i] = i + 1
        
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            
            spf[i * p] = p
            
            if i % p == 0:
                # p is smallest prime factor of i
                p_pow[i * p] = p_pow[i] * p
                p_sigma[i * p] = p_sigma[i] + p_pow[i * p]
                sigma[i * p] = (sigma[i] // p_sigma[i]) * p_sigma[i * p]
                break
            else:
                p_pow[i * p] = p
                p_sigma[i * p] = p + 1
                sigma[i * p] = sigma[i] * (p + 1)
        
        # Time check
        if i % 1000000 == 0:
            if time.time() - start_time > deadline - 20:
                # Reduce N? Can't easily.
                # Just break and use current N?
                # But sigma array is incomplete.
                # We can't break early in sieve.
                # We must finish sieve to have correct sigmas.
                # So we should pick N conservatively.
                pass

    # Pass 2: Count frequencies
    counts = {}
    max_count = 0
    best_M = 0
    LIMIT_M = 10**18
    
    for i in range(1, N + 1):
        m_val = i * sigma[i]
        if m_val <= LIMIT_M:
            if m_val in counts:
                counts[m_val] += 1
                if counts[m_val] > max_count:
                    max_count = counts[m_val]
                    best_M = m_val
            else:
                counts[m_val] = 1
    
    # If no collisions, max_count is 1, best_M is the first one.
    # We need to return a list of k's.
    
    # Pass 3: Collect k's for best_M
    k_list = []
    for i in range(1, N + 1):
        if i * sigma[i] == best_M:
            k_list.append(i)
            
    # Ensure types are correct
    # M is int
    # k is list of ints
    
    return {'M': best_M, 'k': k_list}
