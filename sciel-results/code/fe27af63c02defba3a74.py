"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Precompute primes up to 10^6 for trial division
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    PRIMES = sieve_primes(1000000)
    
    def phi(x):
        """Compute Euler's totient function for a single number using trial division by primes."""
        result = x
        y = x
        for p in PRIMES:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    # Fast phi for small numbers using precomputed sieve
    def compute_phi_sieve(limit):
        phi_arr = list(range(limit + 1))
        for i in range(2, limit + 1):
            if phi_arr[i] == i:  # i is prime
                for j in range(i, limit + 1, i):
                    phi_arr[j] -= phi_arr[j] // i
        return phi_arr
    
    # Strategy:
    # 1. For small ranges (up to ~200k), use sieve phi and scan for longest runs
    # 2. For larger ranges, use random sampling with phi computation
    # 3. Try to find runs of length up to 200
    
    best_start = 1
    best_len = 1
    
    # Phase 1: Sieve-based scan for small numbers
    # We can sieve up to ~200000 in reasonable time
    SIEVE_LIMIT = 200000
    if time_remaining() > 5:
        phi_small = compute_phi_sieve(SIEVE_LIMIT)
        # Scan for longest runs of distinct phi values
        for s in range(1, SIEVE_LIMIT - 200):
            seen = set()
            k = 0
            while s + k <= SIEVE_LIMIT and phi_small[s + k] not in seen:
                seen.add(phi_small[s + k])
                k += 1
            if k > best_len:
                best_len = k
                best_start = s
            # Early stop if we can't beat current best
            if SIEVE_LIMIT - s < best_len:
                break
    
    # Phase 2: Random search in larger ranges
    # Focus on ranges where numbers are likely to have distinct totients
    # Numbers with many small prime factors tend to have smaller phi values
    # Consecutive numbers with different prime factorizations are more likely to have distinct phi
    
    # Use a smarter approach: check windows of size up to 200
    # Sample random starting points and check for distinct phi values
    
    # For efficiency, we'll check windows of various sizes
    # Start with checking for longer runs (higher value)
    
    # We'll use a two-pass approach:
    # Pass 1: Check for runs of length >= 50 (high value targets)
    # Pass 2: Check for runs of length >= 20 (medium value targets)
    
    def check_run(s, max_k, phi_func):
        """Check for longest run of distinct phi values starting at s."""
        seen = set()
        k = 0
        while k < max_k and s + k <= 10**12:
            v = phi_func(s + k)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k
    
    # Phase 2a: Random search for long runs
    # Try different ranges: near powers of 2, near multiples of primorials, etc.
    
    # Strategy: Try random starting points in various decades
    # Numbers in the range [10^a, 10^(a+1)) for various a
    
    random.seed(42)
    
    # Try ranges from 10^3 to 10^12
    for exp in range(3, 13):
        if time_remaining() < 2:
            break
        lower = 10**exp
        upper = min(10**(exp+1), 10**12 + 1)
        
        # Number of samples to try in this range
        # More samples for smaller ranges (more likely to find good runs)
        n_samples = max(100, min(5000, 50000 // (exp - 2)))
        
        for _ in range(n_samples):
            if time_remaining() < 2:
                break
            
            s = random.randint(lower, upper - 200)
            
            # Check for a run starting at s
            k = check_run(s, 200, phi)
            
            if k > best_len:
                best_len = k
                best_start = s
    
    # Phase 2b: Targeted search around known good regions
    # Numbers that are products of consecutive primes (primorial-like)
    # or numbers with diverse prime factorizations
    
    # Try starting points that are just after a highly composite number
    # or in regions with many primes
    
    if time_remaining() > 5:
        # Try around multiples of small primorials
        primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230]
        for prim in primorials:
            if time_remaining() < 2:
                break
            for mult in range(1, min(100, int(10**12 // prim))):
                if time_remaining() < 2:
                    break
                s = prim * mult + 1  # Start just after a multiple of primorial
                if s + 200 > 10**12:
                    break
                k = check_run(s, 200, phi)
                if k > best_len:
                    best_len = k
                    best_start = s
    
    # Phase 3: Local refinement around best found
    if time_remaining() > 3 and best_len > 10:
        # Try nearby starting points around the best found
        for offset in range(-100, 101):
            if time_remaining() < 1:
                break
            s = best_start + offset
            if s < 1 or s + 200 > 10**12:
                continue
            k = check_run(s, 200, phi)
            if k > best_len:
                best_len = k
                best_start = s
    
    # Final verification and return
    # Ensure the result is valid
    if best_start < 1:
        best_start = 1
    if best_start + best_len - 1 > 10**12:
        best_len = max(1, int(10**12 - best_start) + 1)
    if best_len > 200:
        best_len = 200
    
    return {'start': best_start, 'length': best_len}
# EVOLVE-BLOCK-END
