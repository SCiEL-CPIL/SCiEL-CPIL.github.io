"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We search for M with many k such that k*sigma(k)=M.
    # We generate k that are B-smooth.
    # For each k, we compute sigma(k) and M = k*sigma(k).
    # We group M and count distinct k.
    
    # Primes to use for smoothness
    # Increasing the prime set increases the number of candidates.
    # 2,3,5,7,11,13,17,19,23,29,31,37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    limit_k = 10**9  # k must be <= sqrt(M) and M <= 10^18, so k <= 10^9
    
    m_to_ks = defaultdict(list)
    
    # Recursive generator for smooth numbers and their sigma
    # state: idx (index in primes), curr_k, curr_sigma
    # yields: (k, sigma_k)
    
    def gen_smooth(idx, curr_k, curr_sigma):
        if curr_k > limit_k:
            return
        if idx == len(primes):
            # Check time occasionally? No, yield is fast.
            # But we check in the main loop.
            yield curr_k, curr_sigma
            return
        
        p = primes[idx]
        pk = 1
        s_term = 1
        # s_term is sum_{j=0}^{a} p^j
        # curr_k_new = curr_k * p^a
        # curr_sigma_new = curr_sigma * s_term
        
        while curr_k * pk <= limit_k:
            gen_smooth(idx + 1, curr_k * pk, curr_sigma * s_term)
            
            # Prepare for next power
            if pk > limit_k // p:
                break
            pk *= p
            s_term += pk
    
    # We cannot just yield from a nested function in a tight loop without overhead.
    # Better to collect into a list or process directly.
    # Since the number of smooth numbers is finite and not huge, we can iterate.
    
    # Optimization: 
    # The recursive function yields many values. 
    # Let's use a stack or just let the recursion run and append to a list?
    # No, let's just iterate the generator in the main loop.
    
    # However, Python generators have overhead.
    # Let's try a direct recursive function that updates a global/dict.
    
    m_to_ks.clear()
    
    def update_m(idx, curr_k, curr_sigma):
        if curr_k > limit_k:
            return
        if time.time() > deadline - 10:
            return
        
        if idx == len(primes):
            M = curr_k * curr_sigma
            if M <= 10**18:
                m_to_ks[M].append(curr_k)
            return
        
        p = primes[idx]
        pk = 1
        s_term = 1
        
        while curr_k * pk <= limit_k:
            update_m(idx + 1, curr_k * pk, curr_sigma * s_term)
            
            if pk > limit_k // p:
                break
            pk *= p
            s_term += pk
    
    # Start recursion
    update_m(0, 1, 1)
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
    
    # If we didn't find anything better than 3, try to extend primes or limit?
    # The time is the main constraint.
    # If we have time left, we can try a broader search or different primes.
    
    # For now, this should find 5 or more if they exist in the smooth range.
    # If not, we might need to include numbers with a large prime factor.
    # But smooth numbers are the best bet for collisions.
    
    # Verify best solution
    if best_M > 333312:
        # Re-verify k values for best_M to ensure correctness
        # k must be a divisor of M? No, k*sigma(k)=M implies k|M.
        # So we can just check the found ks.
        valid_ks = []
        for k in best_k:
            # Fast sigma check
            if k * _sigma(k) == best_M and k <= math.isqrt(best_M):
                valid_ks.append(k)
        
        if len(valid_ks) >= best_count:
            best_k = valid_ks
        else:
            # Fallback if verification fails (should not happen)
            best_M = 333312
            best_k = [336, 372, 434]
            best_count = 3

    # Ensure types
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    return {'M': best_M, 'k': best_k}

def _sigma(n):
    """Compute sigma(n)."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            term = 1
            pk = 1
            while temp % p == 0:
                temp //= p
                pk *= p
                term += pk
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result
# EVOLVE-BLOCK-END
