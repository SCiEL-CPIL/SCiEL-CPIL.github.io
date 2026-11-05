import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0  # seconds, leaving 5s margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200

    # 1. Generate primes up to 10^6
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]

    # 2. Efficient phi calculation
    def phi(x):
        if x == 1:
            return 1
        result, m = x, x
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    # 3. Helper: find longest run of distinct values in a list
    def longest_distinct_run(vals):
        n = len(vals)
        if n == 0:
            return 0, 0
        max_len = 1
        max_start = 0
        cur_start = 0
        seen = {}
        for i in range(n):
            v = vals[i]
            if v in seen and seen[v] >= cur_start:
                cur_start = seen[v] + 1
            seen[v] = i
            cur_len = i - cur_start + 1
            if cur_len > max_len:
                max_len = cur_len
                max_start = cur_start
        return max_len, max_start

    # 4. Check a window starting at s: compute phi for s..s+MAX_K-1
    #    Return (all_distinct, longest_k, longest_offset)
    def check_window(s):
        phis = []
        for i in range(MAX_K):
            phis.append(phi(s + i))
        if len(set(phis)) == MAX_K:
            return True, MAX_K, 0
        k, offset = longest_distinct_run(phis)
        return False, k, offset

    # 5. Search strategy:
    #    Phase 1: Try a few deterministic candidates near 10^12
    #    Phase 2: Random sampling in [10^11, 10^12 - 200]
    
    best_k = 1
    best_s = 1

    # Phase 1: Deterministic candidates near 10^12
    deterministic_offsets = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
    for offset in deterministic_offsets:
        if time.time() - start_time > LIMIT:
            break
        s = MAX_S - offset
        if s < MAX_K:
            continue
        all_distinct, k, off = check_window(s)
        if all_distinct:
            return {'start': s, 'length': MAX_K}
        if k > best_k:
            best_k = k
            best_s = s + off

    # Phase 2: Random sampling
    rng = random.Random(42)
    low = 10**11
    high = MAX_S - MAX_K

    while time.time() - start_time < LIMIT:
        s = rng.randint(low, high)
        all_distinct, k, off = check_window(s)
        if all_distinct:
            return {'start': s, 'length': MAX_K}
        if k > best_k:
            best_k = k
            best_s = s + off

    # 6. Return best found
    return {'start': best_s, 'length': best_k}
