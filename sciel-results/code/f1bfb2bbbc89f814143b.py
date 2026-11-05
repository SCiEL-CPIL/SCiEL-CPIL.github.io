import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    target = 123457
    
    # Precompute largest prime factor for all numbers up to target using sieve
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (target + 1)
    lpf[1] = 1
    
    for i in range(2, target + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    # Initialize best with trivial splits (a=1, b=target-1 and a=target-1, b=1)
    # P(1) = 1
    best_a = 1
    best_b = target - 1
    best_score = max(lpf[best_a], lpf[best_b])
    
    # Full linear scan
    # We only need to check a from 2 to target // 2
    # because the pair (a, b) is equivalent to (b, a)
    half_target = target // 2
    for a in range(2, half_target + 1):
        b = target - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
        # Optional: early termination if we hit the theoretical minimum?
        # Theoretical min is 2 (if both are powers of 2).
        # 123457 is odd, so a+b=odd. One is even, one is odd.
        # If a is power of 2, b = odd - power of 2.
        # If we find score 2, we can break.
        if best_score == 2:
            break
            
    return [best_a, best_b]
