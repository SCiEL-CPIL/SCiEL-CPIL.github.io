"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    MAX_N = 10000000

    # Precompute smallest prime factor for fast factorization
    spf = np.zeros(MAX_N + 1, dtype=np.int64)
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    # spf[1] = 0, but we handle n=1 separately

    if time.time() > deadline - 10.0:
        return [11, 6]

    def compute_sigma(n):
        """Compute sigma(n) using precomputed SPF."""
        if n <= 0:
            return 0
        if n == 1:
            return 1
        result = 1
        while n > 1:
            p = int(spf[n])
            if p == 0:
                # n is prime
                result *= (n + 1)
                break
            # Compute p^a || n
            power = 1
            sum_powers = 1
            while n % p == 0:
                n //= p
                power *= p
                sum_powers += power
            result *= sum_powers
        return result

    # Best known: sigma(11)=12, sigma(6)=12, |11/6 - 3/2| = 1/6
    best_a, best_b = 11, 6
    best_num = 1  # numerator of error |2a-3b|
    best_den = 12  # denominator 2*b = 2*6 = 12

    # Iterate over small k = |2a - 3b|
    # For each k, a = (3b + k) / 2 or a = (3b - k) / 2
    # Error = k / (2b), so we want large b for small error
    # Check k = 1, 2, 3, ... up to some limit
    
    max_k = 200  # We'll check up to k=200, which gives error <= 200/(2*1) = 100 for small b
                 # But for large b, error is very small
    
    for k in range(1, max_k + 1):
        if time.time() > deadline - 5.0:
            break
        
        # For a = (3b + k) / 2: need 3b + k even, so b must have same parity as k
        # For a = (3b - k) / 2: need 3b - k even, so b must have same parity as k
        
        # Check a = (3b + k) / 2
        # a must be integer: (3b + k) % 2 == 0
        # a must be in [1, MAX_N], a != b
        # sigma(a) == sigma(b)
        
        # b range: a = (3b+k)/2 <= MAX_N => b <= (2*MAX_N - k)/3
        # a >= 1 => 3b + k >= 2 => b >= (2-k)/3, so b >= 1 for k < 2
        # a != b => (3b+k)/2 != b => 3b+k != 2b => b != -k, always true for b>0
        # Also need a > 0, which is satisfied for b >= 1
        
        # For a = (3b - k) / 2:
        # a must be integer: (3b - k) % 2 == 0
        # a must be in [1, MAX_N], a != b
        # a >= 1 => 3b - k >= 2 => b >= (k+2)/3
        # a <= MAX_N => 3b - k <= 2*MAX_N => b <= (2*MAX_N + k)/3
        # a != b => (3b-k)/2 != b => 3b-k != 2b => b != k
        
        # We want to find pairs where sigma(a) == sigma(b)
        # Strategy: for each k, iterate over b in a reasonable range and check
        
        # To be efficient, we should check large b first (smaller error)
        # But we can't check all b up to 10^7 for each k
        
        # Instead, let's check b in a strategic range
        # For k=1: error = 1/(2b), so b=10^7 gives error = 5e-8
        # We need to find ANY pair with sigma(a)==sigma(b)
        
        # Let's try a sampling approach: check b at regular intervals
        # But this might miss pairs
        
        # Better: for each k, iterate b from MAX_N down to 1, but with a step
        # to keep computation time reasonable
        
        # Actually, let's just iterate b for each k, but break early if we find a good pair
        # and the error is already very small
        
        # For efficiency, check b in chunks or with a step
        # Let's check every b, but limit the total number of checks
        
        # Total budget: ~45 seconds for the search
        # For each k, we can check maybe 10000 b values
        # With 200 k values, that's 2 million sigma computations
        # Each sigma computation is O(log n) with SPF, so ~20 operations
        # 2M * 20 = 40M operations, should be fine in ~10-20 seconds
        
        # Let's be more aggressive: check b from MAX_N down, with step 1, but limit per k
        max_b_checks = 50000  # per k value
        
        # For a = (3b + k) / 2
        # b must satisfy (3b + k) % 2 == 0
        # If k is odd, b must be odd; if k is even, b must be even
        
        b_start = MAX_N
        b_end = 1
        step = 1
        
        # Parity constraint: (3b + k) % 2 == 0 => b % 2 == k % 2
        # So if k is odd, start with odd b; if k is even, start with even b
        
        if k % 2 == 1:
            if b_start % 2 == 0:
                b_start -= 1
            # step = 2 to maintain odd
            step = 2
        else:
            if b_start % 2 == 1:
                b_start -= 1
            # step = 2 to maintain even
            step = 2
        
        checks_this_k = 0
        for b in range(b_start, b_end - 1, -step):
            if checks_this_k >= max_b_checks:
                break
            if time.time() > deadline - 3.0:
                break
            
            checks_this_k += 1
            
            # Compute a = (3b + k) / 2
            numerator = 3 * b + k
            if numerator % 2 != 0:
                continue  # Shouldn't happen due to parity constraint, but safe
            a = numerator // 2
            
            if a < 1 or a > MAX_N or a == b:
                continue
            
            # Check sigma(a) == sigma(b)
            sigma_a = compute_sigma(a)
            sigma_b = compute_sigma(b)
            
            if sigma_a == sigma_b:
                # Found a pair! Error = k / (2b)
                # Compare with best: k/(2b) < best_num/best_den
                # i.e., k * best_den < best_num * 2b
                if k * best_den < best_num * 2 * b:
                    best_num = k
                    best_den = 2 * b
                    best_a, best_b = a, b
                    # If error is very small, we can break early
                    # k/(2b) < 1e-6 means we're doing very well
                    if k < 2 * b * 10**(-6):
                        break
        
        # Also check a = (3b - k) / 2
        b_start2 = MAX_N
        # (3b - k) % 2 == 0 => b % 2 == k % 2 (same parity constraint)
        if k % 2 == 1:
            if b_start2 % 2 == 0:
                b_start2 -= 1
            step2 = 2
        else:
            if b_start2 % 2 == 1:
                b_start2 -= 1
            step2 = 2
        
        checks_this_k2 = 0
        for b in range(b_start2, b_end - 1, -step2):
            if checks_this_k2 >= max_b_checks:
                break
            if time.time() > deadline - 3.0:
                break
            
            checks_this_k2 += 1
            
            # Compute a = (3b - k) / 2
            numerator = 3 * b - k
            if numerator < 2:
                break  # a would be < 1, and b is decreasing, so all further b will also fail
            if numerator % 2 != 0:
                continue
            a = numerator // 2
            
            if a < 1 or a > MAX_N or a == b:
                continue
            
            # Check sigma(a) == sigma(b)
            sigma_a = compute_sigma(a)
            sigma_b = compute_sigma(b)
            
            if sigma_a == sigma_b:
                # Found a pair! Error = k / (2b)
                if k * best_den < best_num * 2 * b:
                    best_num = k
                    best_den = 2 * b
                    best_a, best_b = a, b
                    if k < 2 * b * 10**(-6):
                        break

    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
