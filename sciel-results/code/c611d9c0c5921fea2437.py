import time

def construct():
    # Deadline guard
    deadline = time.time() + 80
    
    target = 123457
    
    # Precompute largest prime factor for all numbers up to target
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (target + 1)
    lpf[1] = 1
    
    # Sieve for Largest Prime Factor
    # For each prime i, update lpf for all multiples of i
    for i in range(2, target + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    # Find the split a+b=target that minimizes max(lpf[a], lpf[b])
    # We only need to check a up to target//2 due to symmetry
    best_a = 1
    best_b = target - 1
    best_score = max(lpf[best_a], lpf[best_b])
    
    half_target = target // 2
    
    for a in range(2, half_target + 1):
        if time.time() > deadline:
            break
        b = target - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
