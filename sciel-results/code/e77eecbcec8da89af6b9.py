import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Primes p == 1 mod 4
    # We only need the first few
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397]
    
    LIMIT = 10**14
    
    best_N = 1
    best_score = 0
    best_exponents = {}
    
    # DFS to find N maximizing product of (e_i + 1)
    # We restrict to the first ~10 primes as product grows fast
    def dfs(idx, current_N, current_score):
        nonlocal best_N, best_score, best_exponents
        
        if time.time() > deadline - 5:
            return
            
        if current_score > best_score:
            best_score = current_score
            best_N = current_N
            # We need to track exponents to reconstruct? 
            # Actually, we just need N to generate pairs.
            # But we need to know N.
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Try exponents from high to low? Or low to high?
        # We want to maximize score.
        # Max exponent e such that p^e <= LIMIT // current_N
        max_e = 0
        temp = current_N
        while temp * p <= LIMIT:
            temp *= p
            max_e += 1
            
        # We can try all e from 0 to max_e
        # To be efficient, we might want to try larger e first if p is small?
        # But score increase is (e+1).
        # Let's just iterate all.
        for e in range(max_e + 1):
            new_N = current_N * (p ** e)
            new_score = current_score * (e + 1)
            dfs(idx + 1, new_N, new_score)
            
    dfs(0, 1, 1)
    
    if best_N < 2:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Generate pairs for best_N
    pairs = []
    # a^2 + b^2 = N, a <= b
    # a^2 <= N/2
    max_a = isqrt(best_N // 2)
    
    for a in range(1, max_a + 1):
        if time.time() > deadline - 2:
            break
        rem = best_N - a * a
        b = isqrt(rem)
        if b * b == rem:
            if a <= b:
                pairs.append([a, b])
                
    return {'N': best_N, 'pairs': pairs}
