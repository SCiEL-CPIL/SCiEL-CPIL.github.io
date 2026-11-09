"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

MAX_VAL = 10000000

def _build_spf(n):
    """Build smallest prime factor sieve up to n."""
    spf = list(range(n + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(n)) + 1):
        if spf[i] == i:  # i is prime
            step = i
            start = i * i
            for j in range(start, n + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _sigma_from_spf(n, spf):
    """Compute sigma(n) using precomputed SPF."""
    if n == 1:
        return 1
    result = 1
    while n > 1:
        p = spf[n]
        power_sum = 1
        power = 1
        while n % p == 0:
            n //= p
            power *= p
            power_sum += power
        result *= power_sum
    return result

def construct():
    N = MAX_VAL
    spf = _build_spf(N)
    
    best_err_num = float('inf')
    best_err_den = 1
    best_a = None
    best_b = None
    
    # We want |2a - 3b| / (2b) to be small.
    # This implies |2a - 3b| is small and b is large.
    # Let |2a - 3b| = d.
    # a = (3b +/- d) / 2.
    # We iterate b from a lower bound to N.
    # Lower bound: to beat current best error ~ 2.45e-7, we need b > 1/(2*2.45e-7) ~ 2e6.
    # We'll start from 1e6 to be safe, but 2e6 is faster.
    start_b = 1000000
    
    # Check small d values
    D = 5
    
    for b in range(start_b, N + 1):
        sigma_b = _sigma_from_spf(b, spf)
        
        for d in range(1, D + 1):
            # Check a = (3b + d) / 2
            num1 = 3 * b + d
            if num1 % 2 == 0:
                a1 = num1 // 2
                if 1 <= a1 <= N:
                    if a1 != b:
                        sigma_a1 = _sigma_from_spf(a1, spf)
                        if sigma_a1 == sigma_b:
                            # Error is d / (2b)
                            if d * best_err_den < best_err_num * (2 * b):
                                best_err_num = d
                                best_err_den = 2 * b
                                best_a = a1
                                best_b = b
            
            # Check a = (3b - d) / 2
            num2 = 3 * b - d
            if num2 > 0 and num2 % 2 == 0:
                a2 = num2 // 2
                if 1 <= a2 <= N:
                    if a2 != b:
                        sigma_a2 = _sigma_from_spf(a2, spf)
                        if sigma_a2 == sigma_b:
                            # Error is d / (2b)
                            if d * best_err_den < best_err_num * (2 * b):
                                best_err_num = d
                                best_err_den = 2 * b
                                best_a = a2
                                best_b = b
    
    if best_a is None:
        return [1, 1]
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
