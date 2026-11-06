"""Program: find A, B (3 distinct ints in [1,128]) maximizing |A+B| with pairwise-coprime distinct sums."""
import itertools
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def _sieve(n):
    is_prime = np.ones(n + 1, dtype=bool)
    if n >= 0:
        is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:n + 1:i] = False
    return is_prime

def construct():
    t_start = time.time()
    DEADLINE = 75  # safety margin before 90s kill

    MAX_SUM = 256
    is_prime = _sieve(MAX_SUM)

    def score_of(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        n = len(distinct)
        # Check pairwise coprime
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) > 1:
                    return 0
        return n

    # Verify a known optimal solution: 9 distinct primes => pairwise coprime
    # A = [1, 7, 13], B = [4, 10, 16]
    # Sums: 5,11,17, 11,17,23, 17,23,29
    # Distinct: {5, 11, 17, 23, 29} - all primes, pairwise coprime
    # But wait, we want to maximize |A+B| which is the number of DISTINCT sums
    # Let me find a better solution with 9 distinct sums
    
    # Try: A = [1, 2, 3], B = [4, 5, 6]
    # Sums: 5,6,7, 6,7,8, 7,8,9 -> distinct: {5,6,7,8,9} = 5, not all coprime
    
    # Need 9 distinct sums that are pairwise coprime (all primes)
    # A = [a, a+d1, a+2*d1], B = [b, b+d2, b+2*d2] with gcd(d1,d2)=1
    # Sums form a 3x3 grid: a+b + i*d1 + j*d2 for i,j in {0,1,2}
    # All 9 must be prime
    
    # Known: A=[1,7,13], B=[4,10,16] gives sums 5,11,17,11,17,23,17,23,29
    # Distinct: {5,11,17,23,29} = 5 distinct values
    
    # Let me search for a solution with 9 distinct prime sums
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)

    max_d = 63  # since 2*d <= 127

    # First, try small d1, d2 values which are more likely to have prime clusters
    d1_list = list(range(1, max_d + 1))
    d2_list = list(range(1, max_d + 1))
    
    # Sort by likelihood of finding primes (smaller steps tend to work better)
    for d1 in d1_list:
        if time.time() - t_start > DEADLINE:
            break
        for d2 in d2_list:
            if math.gcd(d1, d2) != 1:
                continue
            if time.time() - t_start > DEADLINE:
                break
            
            a_max = 128 - 2 * d1
            b_max = 128 - 2 * d2
            if a_max < 1 or b_max < 1:
                continue
            
            offsets = np.array([i * d1 + j * d2 for i in range(3) for j in range(3)], dtype=np.int32)
            
            for a in range(1, a_max + 1):
                if time.time() - t_start > DEADLINE:
                    break
                b_vals = np.arange(1, b_max + 1, dtype=np.int32)
                sums = a + b_vals[np.newaxis, :] + offsets[:, np.newaxis]
                
                # Ensure all indices are within bounds
                if sums.max() > MAX_SUM or sums.min() < 0:
                    continue
                
                valid_b_mask = np.all(is_prime[sums], axis=0)
                
                if np.any(valid_b_mask):
                    b_idx = np.argmax(valid_b_mask)
                    b = int(b_vals[b_idx])
                    A = [a, a + d1, a + 2 * d1]
                    B = [b, b + d2, b + 2 * d2]
                    s = score_of(A, B)
                    if s >= best_score:
                        return [A, B]

    # Fallback random search
    import random
    rng = random.Random(42)
    if time.time() - t_start < DEADLINE - 5:
        for _ in range(50000):
            if time.time() - t_start > DEADLINE:
                break
            A = rng.sample(range(1, 129), 3)
            B = rng.sample(range(1, 129), 3)
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
                if s >= 9:
                    return [A, B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
