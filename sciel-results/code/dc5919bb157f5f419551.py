import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # Fast sigma computation using smallest prime factor sieve
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for n in range(2, MAX_N + 1):
        p = spf[n]
        m = n // p
        power = p
        while m > 1 and m % p == 0:
            power *= p
            m //= p
        factor = (power * p - 1) // (p - 1)
        if m == 1:
            sigma[n] = factor
        else:
            sigma[n] = sigma[m] * factor
    
    best_error = float('inf')
    best_a = None
    best_b = None
    
    # d = 2a - 3b, a = (3b + d) / 2
    # Error = |d| / (2b)
    # For each |d| from 1 upward, find largest b with sigma(a) == sigma(b)
    
    max_d = 1000
    
    for abs_d in range(1, max_d + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # If even the best possible error for this |d| is >= best_error, break
        if abs_d / (2 * MAX_N) >= best_error:
            break
        
        for d in [abs_d, -abs_d]:
            if time.time() - start_time > DEADLINE:
                break
            
            # b parity must match d parity (3b + d even => b ≡ d (mod 2))
            b_parity = d % 2
            if b_parity < 0:
                b_parity += 2
            
            b_min = max(1, (2 - d + 2) // 3)
            b_max = min(MAX_N, (2 * MAX_N - d) // 3)
            
            if b_max < b_min:
                continue
            
            if b_max % 2 != b_parity:
                b_max -= 1
            if b_max < b_min:
                continue
            
            # Scan b from b_max downward, step 2
            max_checks = 10000
            checked = 0
            for b in range(b_max, b_min - 1, -2):
                if checked >= max_checks:
                    break
                if time.time() - start_time > DEADLINE:
                    break
                checked += 1
                
                a = (3 * b + d) // 2
                if a < 1 or a > MAX_N:
                    continue
                
                if sigma[a] == sigma[b]:
                    error = abs_d / (2 * b)
                    if error < best_error:
                        best_error = error
                        best_a = a
                        best_b = b
                    break
    
    if best_a is None:
        return [11, 6]
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
