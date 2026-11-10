import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    MAX_N = 10000000
    
    # Linear sieve for sigma(n)
    sigma = [0] * (MAX_N + 1)
    spow = [0] * (MAX_N + 1)
    sval = [0] * (MAX_N + 1)
    is_comp = [False] * (MAX_N + 1)
    primes = []
    
    sigma[1] = 1
    spow[1] = 1
    sval[1] = 1
    
    for i in range(2, MAX_N + 1):
        if not is_comp[i]:
            primes.append(i)
            spow[i] = i
            sval[i] = 1 + i
            sigma[i] = 1 + i
        
        for p in primes:
            j = i * p
            if j > MAX_N:
                break
            is_comp[j] = True
            if i % p == 0:
                spow[j] = spow[i] * p
                sval[j] = sval[i] + spow[j]
                sigma[j] = (sigma[i] // sval[i]) * sval[j]
                break
            else:
                spow[j] = p
                sval[j] = 1 + p
                sigma[j] = sigma[i] * (1 + p)
                
    if time.time() > deadline - 10:
        return [11, 6]
        
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    # Search for better pairs
    # Iterate b from MAX_N down to 1 to find small errors early
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline - 5:
            break
            
        sb = sigma[b]
        target = 1.5 * b
        # We want |a/b - 1.5| < best_err
        # |a - 1.5b| < best_err * b
        max_diff = best_err * b
        
        # Check integers a near 1.5b
        # a must be integer
        # Range: [target - max_diff, target + max_diff]
        low = int(target - max_diff)
        high = int(target + max_diff)
        
        # Clamp to [1, MAX_N]
        if low < 1: low = 1
        if high > MAX_N: high = MAX_N
        
        # To be safe and find the best in the window, 
        # we should check the closest ones first if possible, 
        # but the window is small once best_err is small.
        # If best_err is large, window is large.
        # However, we start with a decent best_err (11/6 -> 0.0833).
        # max_diff for b=10^7 is 833333. That's a huge window!
        # We need to be smarter.
        
        # Instead of scanning the whole window, 
        # we can just check the nearest few integers?
        # No, if best_err is 0.08, we might miss a better one further away?
        # No, we are minimizing error. The closest a to 1.5b has the smallest error.
        # So we only need to check the a's with the smallest |a - 1.5b|.
        # Which are the integers closest to 1.5b.
        
        # So we can just check a in {floor(1.5b), ceil(1.5b), ...}
        # How many?
        # We want |a/b - 1.5| < best_err.
        # The minimum error for any a is dist(1.5b, Z)/b.
        # If dist(1.5b, Z)/b >= best_err, no a in this b can improve.
        # If dist(1.5b, Z)/b < best_err, we check the closest a's.
        
        a0 = int(target)
        a1 = a0 + 1
        # Check a0 and a1 (and maybe neighbors if best_err is very large)
        # But best_err decreases.
        # Let's just check a0-2 to a1+2.
        
        candidates = [a0-2, a0-1, a0, a0+1, a0+2, a1+2]
        # Remove duplicates and out of bounds
        seen = set()
        for a in candidates:
            if a in seen: continue
            seen.add(a)
            if a < 1 or a > MAX_N: continue
            if a == b: continue
            if 2*a == 3*b: continue
            
            if sigma[a] == sb:
                err = abs(a/b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a, best_b = a, b
                    
    return [int(best_a), int(best_b)]
