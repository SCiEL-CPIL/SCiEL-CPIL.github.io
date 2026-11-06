import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24
    
    # Precompute small prime parts
    # small_part[m] = product of p^v_p(m) for p in primes
    small_parts = [1] * (max_val + 1)
    
    # For each prime, multiply into small_parts for its multiples
    # This is not quite right. We need the power.
    # Better to compute for each m individually or use a sieve-like approach.
    
    # Approach:
    # Initialize small_parts[m] = 1
    # For each p in primes:
    #   for m in range(p, max_val+1, p):
    #       # count power of p in m
    #       # this is slow if done naively inside the loop for every multiple
    
    # Optimized:
    # small_parts = [1] * (max_val + 1)
    # for p in primes:
    #     for m in range(p, max_val + 1, p):
    #         temp = m
    #         while temp % p == 0:
    #             small_parts[m] *= p
    #             temp //= p
             
    # This is efficient enough.
    
    for p in primes:
        for m in range(p, max_val + 1, p):
            temp = m
            while temp % p == 0:
                small_parts[m] *= p
                temp //= p
                
    best_n = 1000
    best_score = 25 # Max possible distinct values is 24
    
    start_time = time.time()
    deadline = start_time + 50
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
            
        # Get the 24 small parts
        # Use a set to count distinct values
        # Since 24 is small, a set is fine.
        parts = {small_parts[n + i] for i in range(1, 25)}
        score = len(parts)
        
        if score < best_score:
            best_score = score
            best_n = n
            if best_score == 1:
                break
                
    return best_n
