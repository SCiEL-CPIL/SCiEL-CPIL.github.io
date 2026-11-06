import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    # Sieve of Eratosthenes for primes up to 10^6
    SIEVE_LIMIT = 1000000
    is_prime = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime[i]]
    
    # Smallest prime factor sieve for fast phi computation
    spf = list(range(SIEVE_LIMIT + 1))
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def phi(x):
        """Compute Euler's totient function phi(x)."""
        if x <= SIEVE_LIMIT:
            result = x
            n = x
            while n > 1:
                p = spf[n]
                result -= result // p
                while n % p == 0:
                    n //= p
            return result
        else:
            # For large x, trial division by primes up to sqrt(x)
            result = x
            n = x
            for p in primes:
                if p * p > n:
                    break
                if n % p == 0:
                    result -= result // p
                    while n % p == 0:
                        n //= p
            if n > 1:
                result -= result // n
            return result
    
    def check_run(s, k, max_k=200):
        """Check if phi(s)..phi(s+k-1) are pairwise distinct. Returns length of distinct run."""
        seen = set()
        for i in range(k):
            val = phi(s + i)
            if val in seen:
                return i
            seen.add(val)
            if elapsed() > deadline:
                return i
        return k
    
    def find_best_in_range(start, end, max_k=200):
        """Find the best run in [start, end]."""
        best_k = 1
        best_s = start
        for s in range(start, end + 1):
            if elapsed() > deadline:
                break
            # Quick check: compute phi for a window
            k = check_run(s, max_k)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    break
        return best_k, best_s
    
    # Strategy 1: Exhaustive scan of small range
    best_k, best_s = find_best_in_range(1, 50000)
    
    if elapsed() > deadline - 20:
        return {'start': best_s, 'length': best_k}
    
    # Strategy 2: Targeted starts based on primorials and prime powers
    # Long runs of distinct totients often occur near numbers with many distinct prime factors
    # or near numbers that are products of consecutive primes
    
    # Try starts near primorials
    primorial_starts = []
    p = 2
    prod = 1
    for prime in primes[:20]:
        prod *= prime
        if prod <= 10**12:
            primorial_starts.append(prod)
            primorial_starts.append(prod // 2)
            primorial_starts.append(prod * 2)
            primorial_starts.append(prod - 100)
            primorial_starts.append(prod + 100)
            primorial_starts.append(prod // 3)
            primorial_starts.append(prod * 3)
        else:
            break
    
    # Remove invalid starts
    primorial_starts = [s for s in primorial_starts if 1 <= s <= 10**12 - 200]
    
    for s in sorted(set(primorial_starts)):
        if elapsed() > deadline - 10:
            break
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 200:
                break
        # Also try nearby starts
        for delta in [-10, -5, -1, 1, 5, 10, 20, 50]:
            ns = s + delta
            if 1 <= ns <= 10**12 - 200 and elapsed() < deadline - 5:
                k2 = check_run(ns, 200)
                if k2 > best_k:
                    best_k = k2
                    best_s = ns
                    if best_k >= 200:
                        break
    
    if elapsed() > deadline - 15:
        return {'start': best_s, 'length': best_k}
    
    # Strategy 3: Random sampling across the range
    random.seed(42)
    for _ in range(2000):
        if elapsed() > deadline - 5:
            break
        # Sample from various scales
        scale = random.choice([10**6, 10**7, 10**8, 10**9, 10**10, 10**11])
        s = random.randint(1, min(10**12 - 200, scale * 100))
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 200:
                break
        # Local refinement around good finds
        if k > 10:
            for delta in range(-50, 51, 10):
                ns = s + delta
                if 1 <= ns <= 10**12 - 200 and elapsed() < deadline - 3:
                    k2 = check_run(ns, 200)
                    if k2 > best_k:
                        best_k = k2
                        best_s = ns
                        if best_k >= 200:
                            break
    
    if elapsed() > deadline - 5:
        return {'start': best_s, 'length': best_k}
    
    # Strategy 4: More systematic scan in promising regions
    # Numbers with many small prime factors tend to have "interesting" totients
    # Try scanning around multiples of small primorials
    
    small_primes = [2, 3, 5, 7, 11, 13]
    # Generate some candidate regions
    for m in range(1, 100):
        if elapsed() > deadline - 2:
            break
        base = m * 30030  # 2*3*5*7*11*13
        if base > 10**12 - 200:
            break
        # Scan a window around base
        scan_start = max(1, base - 200)
        scan_end = min(10**12 - 200, base + 200)
        for s in range(scan_start, scan_end + 1, 5):
            if elapsed() > deadline - 1:
                break
            k = check_run(s, 200)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    break
    
    return {'start': best_s, 'length': best_k}
