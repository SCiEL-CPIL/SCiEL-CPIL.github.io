import math
from array import array

MAX_VAL = 10000000

def _build_spf(n):
    """Build smallest prime factor sieve up to n using array for memory efficiency."""
    spf = array('I', range(n + 1))
    for i in range(2, int(math.isqrt(n)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
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
    
    # Iterate b from large to small to find large denominators early
    # Prune if minimum possible error for this b is worse than current best
    for b in range(N, 0, -1):
        # Check pruning condition
        # Min error for this b is 1/(2b) (if k=1 is possible)
        # If 1/(2b) > current_best, no solution for this b or smaller b can be better
        if best_err_num != float('inf'):
            # Use float comparison for speed, it's just for pruning
            if 1.0 / (2.0 * b) > best_err_num / best_err_den:
                break
                
        sig_b = _sigma_from_spf(b, spf)
        
        # Check small k values
        # a = (3b +/- k) / 2
        for k in range(1, 11):
            # Case 1: 2a = 3b + k
            num1 = 3 * b + k
            if num1 % 2 == 0:
                a = num1 // 2
                if 1 <= a <= N:
                    if _sigma_from_spf(a, spf) == sig_b:
                        # Error is k / (2b)
                        # Compare k/(2b) with best_err_num/best_err_den
                        # k * best_err_den < best_err_num * 2b
                        if k * best_err_den < best_err_num * 2 * b:
                            best_err_num = k
                            best_err_den = 2 * b
                            best_a = a
                            best_b = b
                            
            # Case 2: 2a = 3b - k
            num2 = 3 * b - k
            if num2 > 0 and num2 % 2 == 0:
                a = num2 // 2
                if a <= N:
                    if _sigma_from_spf(a, spf) == sig_b:
                        if k * best_err_den < best_err_num * 2 * b:
                            best_err_num = k
                            best_err_den = 2 * b
                            best_a = a
                            best_b = b
                            
    if best_a is None:
        # Fallback, should not happen
        return [1, 1]
        
    return [best_a, best_b]
