"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds, with safety margin we'll use 110s

    def elapsed():
        return time.time() - start_time

    def deadline_exceeded():
        return elapsed() > 110.0

    # Sieve for primes up to a reasonable limit
    # For numbers up to ~10^7, we need primes up to sqrt(10^7) ~ 3162
    # But for faster phi computation, precompute smallest prime factor
    SIEVE_LIMIT = 10**7  # We'll compute phi for numbers up to this range
    
    # Build smallest prime factor (spf) array up to SIEVE_LIMIT
    def build_spf(limit):
        spf = list(range(limit + 1))
        for i in range(2, int(limit**0.5) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i*i, limit + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        return spf

    # For larger numbers, we need trial division with precomputed primes
    def get_primes_up_to(limit):
        sieve = bytearray([1]) * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, limit + 1) if sieve[i]]

    # Precompute spf for small numbers
    SPF_LIMIT = 10**6
    spf = build_spf(SPF_LIMIT)

    def phi_small(x):
        """Compute phi(x) using precomputed spf for x <= SPF_LIMIT."""
        if x <= 1:
            return 1
        result = x
        y = x
        while y > 1:
            p = spf[y]
            result -= result // p
            while y % p == 0:
                y //= p
        return result

    # Get primes up to 10^6 for trial division of larger numbers
    primes_list = get_primes_up_to(10**6)

    def phi_large(x):
        """Compute phi(x) for x > SPF_LIMIT using trial division by primes."""
        if x <= 1:
            return 1
        result = x
        y = x
        for p in primes_list:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def phi(x):
        if x <= SPF_LIMIT:
            return phi_small(x)
        else:
            return phi_large(x)

    # Strategy: search for long runs of consecutive integers with distinct totients
    # We'll scan through ranges and track the maximum run length found
    
    best_k = 1
    best_s = 1

    # First, do a quick scan of a moderate range to get a baseline
    # Then expand search if time permits
    
    # Scan range [1, N] for consecutive runs
    def scan_range(start, end, max_run=200):
        """Scan [start, end] for longest run of consecutive ints with distinct phi values."""
        nonlocal best_k, best_s
        if end - start + 1 < 2:
            return
        # Compute phi values
        n = end - start + 1
        phi_vals = [0] * (n + 1)
        phi_vals[0] = 0
        for i in range(1, n + 1):
            phi_vals[i] = phi(start + i - 1)
        
        # Find longest run
        for s in range(start, min(start + max_run, end + 1)):
            if deadline_exceeded():
                return
            seen = set()
            k = 0
            idx = s - start + 1
            while idx <= n and phi_vals[idx] not in seen:
                seen.add(phi_vals[idx])
                k += 1
                idx += 1
                if k >= max_run:
                    break
            if k > best_k:
                best_k = k
                best_s = s

    # Start with a smaller range to get quick results
    # The seed program found something in [1, 20000], let's start there
    scan_range(1, 50000, 200)

    # If we have time, expand the search
    # Try different ranges
    ranges_to_try = [
        (1, 100000),
        (50000, 150000),
        (100000, 200000),
        (10000, 100000),
        (200000, 300000),
    ]

    for (a, b) in ranges_to_try:
        if deadline_exceeded():
            break
        if a > best_s - 100 and b > best_s + best_k + 100:
            continue  # Already scanned this region
        scan_range(a, b, 200)

    # If still have time, do a broader random-ish search
    # Focus on areas where collisions might be less likely
    # Typically, numbers with many small prime factors have smaller phi,
    # so regions with "smooth" numbers might have more distinct values
    
    # Try searching in larger gaps
    if not deadline_exceeded():
        # Search in [10^5, 10^6] in chunks
        chunk_size = 50000
        pos = 100000
        while pos < 1000000 and not deadline_exceeded():
            scan_range(pos, min(pos + chunk_size, 1000000), 200)
            pos += chunk_size

    # Final safety check
    if deadline_exceeded():
        pass  # Just return what we have

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
