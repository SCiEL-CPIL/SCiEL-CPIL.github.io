import math
import time

def construct():
    target = 123457
    
    # Sieve for Largest Prime Factor (LPF)
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (target + 1)
    lpf[1] = 1
    
    # Initialize for primes
    for i in range(2, target + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    # Find the pair (a, b) with a+b=target that minimizes max(lpf[a], lpf[b])
    best_score = target # Max possible LPF is target
    best_a = 1
    best_b = target - 1
    
    # We only need to check a up to target//2 because (a,b) and (b,a) are symmetric
    for a in range(1, target // 2 + 1):
        b = target - a
        # P(1) = 1 is handled by lpf[1]=1
        score = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
