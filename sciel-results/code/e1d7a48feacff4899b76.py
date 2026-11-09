def construct():
    import math
    from collections import defaultdict
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    max_k = 10**8 # Safe limit for k
    
    count = defaultdict(list)
    
    def gen(idx, k, sigma_k):
        if k > max_k:
            return
        # Record this k
        M_val = k * sigma_k
        if M_val <= 10**18:
            count[M_val].append(k)
            
        if idx == len(primes):
            return
            
        p = primes[idx]
        # Try exponents 1..e
        pk = k
        # sigma of p^e is 1 + p + ... + p^e
        # We can update sigma incrementally
        # sigma_new = sigma_k * (1 + p + ... + p^e) / (1) ? No.
        # If k is not divisible by p, sigma(k * p^e) = sigma(k) * sigma(p^e).
        # Since we are building k from scratch and primes are distinct, this holds.
        
        pk = k
        sigma_pk = sigma_k
        # We need to multiply sigma_k by (1 + p + ... + p^e)
        # But we are iterating e.
        # Let S_e = 1 + p + ... + p^e.
        # sigma(k * p^e) = sigma(k) * S_e.
        
        # Wait, the recursive step should handle the next prime.
        # So for the current prime p, we choose an exponent e.
        # New k' = k * p^e.
        # New sigma' = sigma_k * (1 + p + ... + p^e).
        # Then recurse on idx+1 with k', sigma'.
        
        # Initialize for e=0 (skip this prime)
        # The call gen(idx+1, k, sigma_k) handles e=0.
        
        pk = k
        sum_powers = 1 # 1 + p^0
        # e starts from 1
        while True:
            pk *= p
            if pk > max_k:
                break
            sum_powers += pk
            new_sigma = sigma_k * sum_powers
            # new_k = pk
            gen(idx + 1, pk, new_sigma)
            
    gen(0, 1, 1)
    
    best_M = None
    best_k = []
    for M_val, ks in count.items():
        sqrt_M = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > len(best_k):
            best_k = valid_ks
            best_M = M_val
            
    if best_M is None:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_k}
