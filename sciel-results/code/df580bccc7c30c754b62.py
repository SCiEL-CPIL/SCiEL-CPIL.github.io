import time
import math

def construct():
    start_time = time.time()
    MAX_N = 1000000
    MIN_N = 10000
    
    # Time budget guard - safety margin before 90s kill
    DEADLINE = 85.0  # Seconds
    
    N_LIMIT = MAX_N + 1
    
    # 1. Compute Smallest Prime Factor (SPF) using a standard sieve
    spf = list(range(N_LIMIT + 1))
    
    for i in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # 2. Compute powerful part for each number
    # p_part[i] holds the product of p^e for all prime factors p of i with exponent e >= 2
    p_part = [1] * (N_LIMIT + 1)
    
    for i in range(2, N_LIMIT + 1):
        x = i
        res = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                res *= (p ** e)
        p_part[i] = res
        
    # 3. Search for max product
    # We only care about n where both P(n) > 1 and P(n+1) > 1 to save time
    best_n = MIN_N
    max_score = 0
    
    # Check time before search
    if time.time() - start_time > DEADLINE:
        return best_n
        
    for n in range(MIN_N, MAX_N + 1):
        # Optimization: skip if either part is 1 (product will be 1, unlikely to be max unless all are 1)
        # But we must be careful: what if the max score is 1? 
        # However, there exist numbers with powerful parts > 1. 
        # If we find any score > 0, we can ignore 1s. 
        # Actually, if p_part[n] == 1 or p_part[n+1] == 1, the product is 1.
        # Since we start with max_score = 0, if we find a 1, it updates.
        # But if we find a 100, it updates.
        # So we can safely skip if either is 1 ONLY IF we are confident a non-1 product exists.
        # Given the range [10000, 1000000], there are definitely powerful numbers.
        # However, to be absolutely safe and simple, we just check the product.
        
        # A more efficient check:
        if p_part[n] > 1 and p_part[n+1] > 1:
            score = p_part[n] * p_part[n+1]
            if score > max_score:
                max_score = score
                best_n = n
                
        # Time check inside the loop
        if time.time() - start_time > DEADLINE:
            break
            
    return best_n
