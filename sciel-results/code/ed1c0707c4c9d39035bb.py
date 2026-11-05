import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    # 1. Generate primes up to 1000 (sqrt(1000000))
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Helper to find largest prime factor
    def largest_prime_factor(x):
        if x <= 1:
            return 1
        res = 1
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                res = p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            res = temp
        return res

    # 3. Search for the best n
    best_n = 10000
    best_val = float('inf')
    
    # We can start from 10000.
    # Note: n(n+1) smoothness is rare. 
    # Trial division is fast enough for 10^6 iterations.
    
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
            
        lpf_n = largest_prime_factor(n)
        lpf_np1 = largest_prime_factor(n + 1)
        
        current_val = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END
