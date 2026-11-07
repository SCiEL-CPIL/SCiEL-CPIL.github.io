"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 10000000
    
    # Step 1: Compute smallest prime factor (SPF) using linear sieve
    spf = [0] * (N + 1)
    spf[1] = 1
    primes = []
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i]:
                break
            if i * p > N:
                break
            spf[i * p] = p
    
    # Step 2: Compute sigma(n) using multiplicative property
    # For each n, decompose into p^k * m where p = spf[n]
    p_power = [1] * (N + 1)
    rest = [1] * (N + 1)
    sigma_pp = [0] * (N + 1)  # sigma of the p^k part
    sigma = [0] * (N + 1)
    sigma[1] = 1
    sigma_pp[1] = 1
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # n = p * m, m divisible by p
            pp_m = p_power[m]
            p_power[n] = p * pp_m
            rest[n] = rest[m]
            sigma_pp[n] = sigma_pp[m] + p_power[n]
        else:
            # n = p * m, m not divisible by p
            p_power[n] = p
            rest[n] = m
            sigma_pp[n] = 1 + p
        sigma[n] = sigma_pp[n] * sigma[rest[n]]
    
    # Step 3: For each b, check the closest integers to 1.5*b
    # a/b ≈ 3/2 means a ≈ 1.5*b
    # We check a = floor(1.5*b) and a = ceil(1.5*b)
    # Using exact integer arithmetic: 1.5*b = 3b/2
    best_error_num = None
    best_error_den = None
    best_a = None
    best_b = None
    
    for b in range(1, N + 1):
        num = 3 * b
        a1 = num // 2
        a2 = a1 + 1 if num % 2 != 0 else a1
        
        for a in (a1, a2):
            if a < 1 or a > N:
                continue
            if a == b:
                continue
            if 2 * a == 3 * b:
                continue
            if sigma[a] != sigma[b]:
                continue
            
            err_num = abs(2 * a - 3 * b)
            err_den = 2 * b
            
            if best_error_num is None:
                best_error_num = err_num
                best_error_den = err_den
                best_a = a
                best_b = b
            else:
                # Compare err_num/err_den < best_error_num/best_error_den
                if err_num * best_error_den < best_error_num * err_den:
                    best_error_num = err_num
                    best_error_den = err_den
                    best_a = a
                    best_b = b
    
    if best_a is None:
        # Fallback: return a valid pair
        return [11, 6]
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
