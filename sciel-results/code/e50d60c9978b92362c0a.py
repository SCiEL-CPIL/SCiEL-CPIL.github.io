import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    N = 123457
    
    # Precompute Largest Prime Factor (LPF) for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1  # P(1) = 1
    
    # Sieve: for each prime p, mark all multiples with p as largest prime factor
    # Since we iterate p in increasing order, the last assignment gives the largest prime factor
    for p in range(2, N + 1):
        if lpf[p] == 0:
            # p is prime
            for j in range(p, N + 1, p):
                lpf[j] = p
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return [1, N - 1]
    
    # Search for optimal a in [1, N//2]
    best_score = float('inf')
    best_a = 1
    best_b = N - 1
    mid = N // 2
    
    for a in range(1, mid + 1):
        b = N - a
        score = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
    
    return [best_a, best_b]
