import math
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def now():
        return time.time()

    # Precompute primes up to 10^6 for trial division of numbers up to 10^12
    LIMIT = 10**6
    is_prime = bytearray(b'\x01') * (LIMIT + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, LIMIT + 1) if is_prime[i]]

    def phi(x):
        if x <= 1:
            return 1
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def find_best_run(s, length_limit=200):
        """Find the longest run of distinct phi values starting at s."""
        seen = set()
        k = 0
        for i in range(s, s + length_limit):
            if i > 10**12:
                break
            v = phi(i)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    def search_range(start, end, step=1):
        """Search for best run in [start, end] with given step."""
        best_k = 0
        best_s = start
        s = start
        while s <= end and now() < DEADLINE:
            k = find_best_run(s)
            if k > best_k:
                best_k = k
                best_s = s
            # Skip ahead: if we found a run of length k, the next possible
            # better run starts after the collision point
            s += max(1, step)
        return best_k, best_s

    # Strategy: search in multiple regions
    # 1. Small range for baseline
    # 2. Medium ranges
    # 3. Large ranges near 10^12
    
    best_k = 1
    best_s = 1

    # Phase 1: Quick search in small range to get a baseline
    if now() < DEADLINE:
        bk, bs = search_range(1, 50000, step=1)
        if bk > best_k:
            best_k = bk
            best_s = bs

    # Phase 2: Search in medium ranges with larger steps
    if now() < DEADLINE:
        regions = [
            (10**6, 10**6 + 10**6),
            (10**7, 10**7 + 10**6),
            (10**8, 10**8 + 10**6),
            (10**9, 10**9 + 10**6),
        ]
        for start, end in regions:
            if now() >= DEADLINE:
                break
            bk, bs = search_range(start, end, step=100)
            if bk > best_k:
                best_k = bk
                best_s = bs

    # Phase 3: Search near 10^12 with larger steps
    if now() < DEADLINE:
        large_start = 10**12 - 10**7
        bk, bs = search_range(large_start, 10**12, step=1000)
        if bk > best_k:
            best_k = bk
            best_s = bs

    # Phase 4: If time remains, do finer search around the best found
    if now() < DEADLINE - 10:
        # Search around the best position with step 1
        search_start = max(1, best_s - 5000)
        search_end = min(10**12, best_s + 5000)
        bk, bs = search_range(search_start, search_end, step=1)
        if bk > best_k:
            best_k = bk
            best_s = bs

    # Phase 5: If still time, try a few more random regions
    if now() < DEADLINE - 5:
        import random
        random.seed(42)
        for _ in range(20):
            if now() >= DEADLINE - 5:
                break
            rand_start = random.randint(10**9, 10**12 - 10**6)
            bk, bs = search_range(rand_start, rand_start + 10**6, step=500)
            if bk > best_k:
                best_k = bk
                best_s = bs

    # Final verification: make sure the result is valid
    # Check that s + k - 1 <= 10^12
    if best_s + best_k - 1 > 10**12:
        best_k = int(10**12) - best_s + 1
        if best_k < 1:
            best_k = 1
            best_s = 1
    if best_k > 200:
        best_k = 200

    return {'start': best_s, 'length': best_k}
