"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    MAX_N = 10**12
    
    def sieve_primes(limit):
        """Generate primes up to limit using sieve."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute small primes for trial division
    SMALL_PRIME_LIMIT = 200000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)
    
    def is_prime(n):
        """Deterministic Miller-Rabin for n < 3.3e24."""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def phi_single(x):
        """Compute Euler's totient function phi(x) exactly."""
        if x == 1:
            return 1
        result = x
        y = x
        for p in small_primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def compute_phi_range(start, end):
        """Compute phi for all integers in [start, end] using sieve method."""
        length = end - start + 1
        if length <= 0:
            return []
        
        # Initialize phi values
        phi_vals = list(range(start, end + 1))
        
        # Sieve: for each prime p, for each multiple m of p in [start,end]:
        # phi[m] -= phi[m] // p
        max_p = int(math.isqrt(end)) + 1
        for p in small_primes:
            if p > max_p:
                break
            first = ((start + p - 1) // p) * p
            for m in range(first, end + 1, p):
                idx = m - start
                if phi_vals[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
        
        # Handle primes: if phi_vals[i] == n and n > 1, n is prime
        for i in range(length):
            n = start + i
            if n == 1:
                phi_vals[i] = 1
            elif phi_vals[i] == n:
                phi_vals[i] = n - 1
        
        return phi_vals
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Dense search using sieve in small-to-medium ranges
    search_ranges = [
        (1, 200000),
        (200000, 1000000),
        (1000000, 5000000),
    ]
    
    for lo, hi in search_ranges:
        if time.time() > deadline - 20:
            break
        
        chunk_size = 200000
        for chunk_start in range(lo, hi + 1, chunk_size):
            if time.time() > deadline - 20:
                break
            chunk_end = min(chunk_start + chunk_size - 1, hi)
            
            phi_vals = compute_phi_range(chunk_start, chunk_end)
            
            # Scan for longest run of distinct phi values
            seen = {}
            current_run_start = chunk_start
            for i, v in enumerate(phi_vals):
                n = chunk_start + i
                if v in seen:
                    run_len = n - current_run_start
                    if run_len > best_k:
                        best_k = run_len
                        best_s = current_run_start
                    current_run_start = n + 1
                    # Periodically clean up seen dict
                    if len(seen) > 50000:
                        seen = {k: pos for k, pos in seen.items() if pos >= current_run_start}
                seen[v] = n
            
            final_run_len = chunk_end - current_run_start + 1
            if final_run_len > best_k:
                best_k = final_run_len
                best_s = current_run_start
    
    # Phase 2: Sparse search in larger ranges using phi_single
    if time.time() < deadline - 25:
        ranges = [
            (10**6, 10**7),
            (10**7, 10**8),
            (10**8, 10**9),
            (10**9, 10**10),
            (10**10, 10**11),
            (10**11, 10**12),
        ]
        
        for lo, hi in ranges:
            if time.time() > deadline - 10:
                break
            num_samples = 40
            for _ in range(num_samples):
                if time.time() > deadline - 5:
                    break
                s = random.randint(lo, min(hi, int(MAX_N) - 200))
                seen = set()
                k = 0
                for i in range(200):
                    n = s + i
                    v = phi_single(n)
                    if v in seen:
                        break
                    seen.add(v)
                    k += 1
                if k > best_k:
                    best_k = k
                    best_s = s
    
    # Phase 3: Local search around best found
    if best_k > 5 and time.time() < deadline - 15:
        window = 2000
        for offset in range(-window, window + 1):
            if time.time() > deadline - 5:
                break
            s = best_s + offset
            if s < 1 or s + 199 > MAX_N:
                continue
            seen = set()
            k = 0
            for i in range(200):
                n = s + i
                v = phi_single(n)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
