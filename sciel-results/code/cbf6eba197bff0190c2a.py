"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

# Precompute smallest prime factor up to 20000 (max sum is 2*10000)
MAX_VAL = 10000
MAX_SUM = 2 * MAX_VAL

def _build_spf(limit):
    spf = list(range(limit + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

_SPF = _build_spf(MAX_SUM)

def _factorize(n):
    """Return set of prime factors of n using SPF table."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = _SPF[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _count_distinct_primes(arr):
    """Count distinct primes dividing sums of all pairs of distinct elements."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= _factorize(s)
    return len(primes)

def _evaluate(arr):
    """Return (num_primes, prime_set) for the array."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= _factorize(s)
    return len(primes), primes

def construct():
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    best_arr = list(range(1, 13))
    best_score = _count_distinct_primes(best_arr)
    
    n = 12
    rng = random.Random(42)
    
    # Strategy 1: Try to find 12 numbers where all pairwise sums are powers of 2
    # If all sums are powers of 2, the only prime factor is 2.
    # Sum of two distinct elements = 2^k
    # Try: pick numbers from {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}
    # But sums of two powers of 2 are not generally powers of 2.
    # Better: use numbers like 2^a - 2^b so that sums work out.
    # Actually, if we use numbers of form 2^k, sum of 2^a + 2^b = 2^a(1+2^{b-a})
    # This is a power of 2 only if a=b, but elements must be distinct.
    # So pure powers of 2 won't work for score 1.
    
    # Strategy 2: Local search with SPF-based fast evaluation
    # Start from a good initial guess and do hill-climbing / simulated annealing
    
    # Good initial: try numbers that are all congruent mod small primes
    # to limit the prime factors of sums
    
    # Let's try: all numbers odd => all sums even => 2 is always a factor
    # If we can make all sums powers of 2 times a small set of primes...
    
    # Try initial: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 (odd numbers)
    # Sums are all even. Let's check what primes appear.
    init_odd = list(range(1, 25, 2))[:12]
    score_odd = _count_distinct_primes(init_odd)
    if score_odd < best_score:
        best_arr = init_odd[:]
        best_score = score_odd
    
    # Try: multiples of a small number
    # All multiples of 3: sums are multiples of 6
    init_mult3 = [3*i for i in range(1, 13)]
    score_m3 = _count_distinct_primes(init_mult3)
    if score_m3 < best_score:
        best_arr = init_mult3[:]
        best_score = score_m3
    
    # Try: numbers of form 2^k * m where m is small
    # Let's try powers of 2 times 1: 1,2,4,8,16,32,64,128,256,512,1024,2048
    init_pow2 = [1 << i for i in range(12)]
    score_p2 = _count_distinct_primes(init_pow2)
    if score_p2 < best_score:
        best_arr = init_pow2[:]
        best_score = score_p2
    
    # Try: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # Sums: 1+2=3, 1+4=5, 2+4=6=2*3, 1+8=9=3^2, etc.
    # Many primes will appear.
    
    # Better initial: try to minimize by using numbers close together
    # or with special structure
    
    # Local search: hill climbing with random restarts
    def _local_search(arr, max_steps=20000):
        """Hill climbing: try replacing one element at a time."""
        current = arr[:]
        current_score, _ = _evaluate(current)
        n = len(current)
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return current, current_score
            improved = False
            # Try swapping one element
            idx = rng.randint(0, n - 1)
            old_val = current[idx]
            # Try a range of new values
            for _ in range(50):
                if time.time() > deadline:
                    return current, current_score
                new_val = rng.randint(1, MAX_VAL)
                if new_val == old_val:
                    continue
                if new_val in current:
                    continue
                current[idx] = new_val
                new_score, _ = _evaluate(current)
                if new_score < current_score:
                    current_score = new_score
                    steps += 1
                    improved = True
                    break
                else:
                    current[idx] = old_val
            if not improved:
                # Try a different element
                pass
            steps += 1
            if steps % 1000 == 0 and time.time() > deadline:
                return current, current_score
        return current, current_score
    
    # Run local search from multiple starting points
    start_configs = [
        list(range(1, 13)),
        list(range(1, 25, 2))[:12],
        [3*i for i in range(1, 13)],
        [1 << i for i in range(12)],
        [2*i+1 for i in range(12)],
        [4*i+1 for i in range(12)],
        [5*i+1 for i in range(12)],
        [6*i+1 for i in range(12)],
    ]
    
    for start in start_configs:
        if time.time() > deadline:
            break
        # Ensure all distinct and <= MAX_VAL
        if len(set(start)) != n or any(x > MAX_VAL or x < 1 for x in start):
            continue
        s, _ = _evaluate(start)
        if s < best_score:
            best_arr = start[:]
            best_score = s
        # Local search
        result, score = _local_search(start, max_steps=5000)
        if score < best_score:
            best_arr = result[:]
            best_score = score
    
    # Additional: try to find a set where all pairwise sums have only primes {2, 3}
    # This would give score 2.
    # Sums must be of form 2^a * 3^b
    # Let's try to construct such a set.
    # If all numbers are odd, sums are even.
    # If all numbers ≡ 1 mod 3, sums ≡ 2 mod 3, so 3 doesn't divide sum.
    # If all numbers ≡ 1 mod 6, sums ≡ 2 mod 6 = 2 mod 6, so sum = 2*(odd not div by 3)
    # Hmm, this is getting complex. Let's just rely on local search.
    
    # One more strategy: try numbers of form 2^k
    # Actually, let's try: 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
    # These are 2^k - 1. Sum of (2^a-1) + (2^b-1) = 2^a + 2^b - 2
    # = 2(2^{a-1} + 2^{b-1} - 1). The term in parens is odd.
    # So sum = 2 * odd. The odd part could have many prime factors.
    
    # Try: 2, 6, 14, 30, 62, 126, 254, 510, 1022, 2046, 4094, 8190
    # = 2*(2^k - 1). Sum = 2*(2^a-1) + 2*(2^b-1) = 2*(2^a + 2^b - 2) = 4*(2^{a-1} + 2^{b-1} - 1)
    # Still complex.
    
    # Final: run a few more local search iterations if time permits
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline:
                break
            # Perturb best solution
            perturbed = best_arr[:]
            for _ in range(3):
                idx = rng.randint(0, n - 1)
                perturbed[idx] = rng.randint(1, MAX_VAL)
            # Ensure distinct
            if len(set(perturbed)) != n or any(x > MAX_VAL or x < 1 for x in perturbed):
                continue
            result, score = _local_search(perturbed, max_steps=3000)
            if score < best_score:
                best_arr = result[:]
                best_score = score
    
    return best_arr
# EVOLVE-BLOCK-END
