import math

def construct():
    # Generate primes up to sqrt(10^12) = 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    W = 200
    max_limit = 10**12
    
    # Try multiple starting positions near the upper bound
    # Use a deterministic step to explore different regions
    base_start = max_limit - W + 1
    
    for attempt in range(500):
        start = base_start - attempt * W
        
        if start < 1:
            break
            
        # Segmented sieve for phi
        nums = [start + i for i in range(W)]
        phis = [start + i for i in range(W)]
        
        for p in primes:
            if p * p > start + W - 1:
                break
                
            # Find first multiple of p in [start, start+W-1]
            m = ((start + p - 1) // p) * p
            
            if m > start + W - 1:
                continue
                
            for x in range(m, start + W, p):
                idx = x - start
                if nums[idx] % p == 0:
                    phis[idx] -= phis[idx] // p
                    while nums[idx] % p == 0:
                        nums[idx] //= p
                        
        # Handle remaining prime factors
        for i in range(W):
            if nums[i] > 1:
                phis[i] -= phis[i] // nums[i]
                
        # Check distinctness
        if len(set(phis)) == W:
            return {'start': start, 'length': W}
            
    # Fallback: try smaller lengths
    for k in range(199, 0, -1):
        for attempt in range(100):
            start = base_start - attempt * k
            if start < 1:
                break
                
            nums = [start + i for i in range(k)]
            phis = [start + i for i in range(k)]
            
            for p in primes:
                if p * p > start + k - 1:
                    break
                    
                m = ((start + p - 1) // p) * p
                if m > start + k - 1:
                    continue
                    
                for x in range(m, start + k, p):
                    idx = x - start
                    if nums[idx] % p == 0:
                        phis[idx] -= phis[idx] // p
                        while nums[idx] % p == 0:
                            nums[idx] //= p
                            
            for i in range(k):
                if nums[i] > 1:
                    phis[i] -= phis[i] // nums[i]
                    
            if len(set(phis)) == k:
                return {'start': start, 'length': k}
                
    return {'start': 1, 'length': 1}
