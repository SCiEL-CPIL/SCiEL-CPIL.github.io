import time
import math
from heapq import heappush, heappop

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    target = 123457
    
    # Fallback solution
    best_a, best_b = 1, target - 1
    
    # Compute largest prime factor for all n up to target
    # P[n] will store the largest prime factor of n
    P = [0] * (target + 1)
    P[1] = 1
    
    # Sieve for largest prime factor
    # Iterate p from 2 to target
    # If P[p] is 0, p is prime
    # Then for all multiples j of p, P[j] = p
    # Since we iterate p in increasing order, the last prime factor assigned to j is the largest one.
    for p in range(2, target + 1):
        if P[p] == 0:
            for j in range(p, target + 1, p):
                P[j] = p
                
    # Now find a, b such that a+b=target and max(P[a], P[b]) is minimized
    min_max_p = float('inf')
    best_a = 1
    best_b = target - 1
    
    for a in range(1, target):
        b = target - a
        current_max = P[a] if P[a] > P[b] else P[b]
        if current_max < min_max_p:
            min_max_p = current_max
            best_a = a
            best_b = b
            # If we found 1, it's the absolute minimum possible (since P(1)=1, P(n)>=1)
            # But P(a) and P(b) must be >= 1. 
            # If a=1, P(a)=1. If b is prime, P(b)=b. max=b.
            # If we find a case where max is small, we keep it.
            # We can break if min_max_p == 1? No, P(a)>=1, P(b)>=1.
            # The minimum possible value for max(P(a), P(b)) is 1 only if a=1 and b=1, which implies target=2.
            # For target > 2, at least one of a,b is > 1, so its P is >= 2.
            # So min_max_p >= 2.
            
    return [best_a, best_b]
