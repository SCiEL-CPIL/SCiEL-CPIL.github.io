import time

def construct():
    """
    Finds n in [10000, 1000000] that minimizes max(LPF(n), LPF(n+1)).
    Uses a sieve to precompute Largest Prime Factors.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1
    
    # Precompute Largest Prime Factor (LPF) for all numbers up to MAX_VAL
    # lpf[i] will hold the largest prime factor of i
    lpf = [0] * (MAX_VAL + 1)
    
    # Sieve of Eratosthenes variant for LPF
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            # p is prime
            for j in range(p, MAX_VAL + 1, p):
                lpf[j] = p
                
    best_n = LO
    best_val = float('inf')
    
    # Search for the minimum value
    for n in range(LO, HI + 1):
        # Check time limit
        if time.time() > deadline:
            break
            
        v = lpf[n]
        v_next = lpf[n+1]
        current_val = v if v > v_next else v_next
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
            # If we hit 2, it's the theoretical minimum for any n>1
            # But consecutive powers of 2 don't exist for n>1.
            # So we can't break early on 2 unless we are sure.
            # We can break if best_val == 2.
            if best_val == 2:
                break

    return best_n
