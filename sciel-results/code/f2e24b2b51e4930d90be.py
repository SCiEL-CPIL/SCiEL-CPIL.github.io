import time
import math
import numpy as np
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0

    # Known good pair: sigma(122)=sigma(80)=186, |122/80 - 3/2| = 1/40
    best_error = Fraction(1, 40)
    best_a, best_b = 122, 80

    def check_pair(a, b):
        nonlocal best_error, best_a, best_b
        if a <= 0 or b <= 0 or a == b:
            return
        num = abs(2 * a - 3 * b)
        if num == 0:
            return
        den = 2 * b
        if num * best_error.denominator < best_error.numerator * den:
            best_error = Fraction(num, den)
            best_a, best_b = a, b

    # Phase 1: Fast sigma sieve up to moderate bound
    L1 = 2_000_000
    sigma = np.zeros(L1 + 1, dtype=np.int64)
    for i in range(1, L1 + 1):
        sigma[i::i] += i

    if time.time() - start_time > deadline - 20:
        return [best_a, best_b]

    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, L1 + 1):
        groups[int(sigma[n])].append(n)

    if time.time() - start_time > deadline - 15:
        return [best_a, best_b]

    # Phase 2: For each group, find pairs with ratio close to 3/2
    for sig_val, members in groups.items():
        if time.time() - start_time > deadline - 5:
            break
        if len(members) < 2:
            continue

        m = len(members)
        # For small groups, check all pairs
        if m <= 200:
            for i in range(m):
                a_val = members[i]
                target_b = 2.0 * a_val / 3.0
                # Binary search
                lo, hi = 0, m - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if members[mid] < target_b:
                        lo = mid + 1
                    else:
                        hi = mid
                for c_idx in range(max(0, lo - 3), min(m, lo + 4)):
                    b_val = members[c_idx]
                    check_pair(a_val, b_val)
        else:
            # For large groups, sample
            step = max(1, m // 500)
            for idx in range(0, m, step):
                a_val = members[idx]
                target_b = 2.0 * a_val / 3.0
                lo, hi = 0, m - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if members[mid] < target_b:
                        lo = mid + 1
                    else:
                        hi = mid
                for c_idx in range(max(0, lo - 5), min(m, lo + 6)):
                    b_val = members[c_idx]
                    check_pair(a_val, b_val)

    if time.time() - start_time > deadline - 10:
        return [best_a, best_b]

    # Phase 3: Targeted generation - for each a, check if sigma(2a/3) or nearby matches sigma(a)
    # This is a direct ratio-based search
    for a in range(1, L1 + 1):
        if time.time() - start_time > deadline - 3:
            break
        sa = int(sigma[a])
        # Target b = 2a/3
        b_target = (2 * a) // 3
        # Check b values near 2a/3
        for b in range(max(1, b_target - 20), min(L1, b_target + 21)):
            if b == a:
                continue
            if int(sigma[b]) == sa:
                check_pair(a, b)
        # Also try b = 2a/3 rounded differently
        b_target2 = (2 * a + 1) // 3
        if b_target2 != b_target:
            for b in range(max(1, b_target2 - 10), min(L1, b_target2 + 11)):
                if b == a:
                    continue
                if int(sigma[b]) == sa:
                    check_pair(a, b)

    if time.time() - start_time > deadline - 5:
        return [best_a, best_b]

    # Phase 4: Meet-in-the-middle factor generation for larger numbers
    # Generate numbers with specific factor structures that give ratio ~3/2
    # Key: if a = 3^k * rest_a and b = 2^j * rest_b, we need sigma(a)=sigma(b)
    # and a/b ≈ 3/2
    
    # Generate prime powers
    max_p = 100_000
    primes = []
    is_prime = np.ones(max_p + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_p)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i in range(2, max_p + 1) if is_prime[i]]

    if time.time() - start_time > deadline - 2:
        return [best_a, best_b]

    # For each prime p, generate p^k and sigma(p^k) = (p^(k+1)-1)/(p-1)
    # We want to build numbers a and b such that sigma(a)=sigma(b) and a/b≈3/2
    # Strategy: enumerate "core" structures and combine
    
    # Simpler targeted approach: check specific ratio patterns
    # a/b ≈ 3/2 means a ≈ 1.5b, so a = 3k, b = 2k approximately
    # sigma(3k) = sigma(2k) when?
    
    # Check pairs where a = 3*x, b = 2*x for x up to L1/3
    for x in range(1, L1 // 3 + 1):
        if time.time() - start_time > deadline - 1:
            break
        a_val = 3 * x
        b_val = 2 * x
        if a_val <= L1 and b_val <= L1:
            if int(sigma[a_val]) == int(sigma[b_val]):
                check_pair(a_val, b_val)

    return [int(best_a), int(best_b)]
