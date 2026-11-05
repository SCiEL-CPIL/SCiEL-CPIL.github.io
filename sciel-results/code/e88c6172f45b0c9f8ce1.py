import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    target = 123457
    
    # Precompute largest prime factor for all numbers up to target
    # Using a sieve approach
    lpf = [1] * (target + 1)  # lpf[i] = largest prime factor of i
    for i in range(2, target + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
    
    # Now find a, b with a + b = target minimizing max(lpf[a], lpf[b])
    # lpf[1] = 1 (by definition P(1)=1)
    best_score = float('inf')
    best_a, best_b = 1, target - 1
    
    for a in range(1, target // 2 + 1):
        b = target - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a, best_b = a, b
            if best_score == 2:
                # Can't do better than 2 (since target is odd, one of a,b must be odd)
                # Actually if target is odd, one is even, one is odd
                # If the odd one is prime, P=that prime. If even one is power of 2, P=2
                # So min possible is min(odd_prime, 2) but odd_prime >= 3
                # So if we find score=2, that means both have lpf <= 2, i.e., both powers of 2
                # But sum of two powers of 2 can't be odd (unless one is 1=2^0)
                # 1 + 123456: lpf[1]=1, lpf[123456]=? 123456 = 2^6 * 1931, so lpf=1931
                # So score=2 is impossible for odd target
                break
    
    # Check if we have time, but this should be very fast
    if time.time() > deadline:
        # Return best found so far
        pass
    
    return [best_a, best_b]
