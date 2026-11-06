import math
import time
import random
from functools import lru_cache
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 145
    max_val = 10**18

    # First 16 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    nprimes = len(primes)

    # Precompute prime powers: prime_powers[i][j] = primes[i]^j
    prime_powers = []
    for p in primes:
        pw = [1]
        while pw[-1] * p <= max_val:
            pw.append(pw[-1] * p)
        prime_powers.append(pw)

    # ---- Fast score using bitmask for prime-exponent tracking ----
    def tau_perp(n):
        if n < 2:
            return 0
        # Factorize n into (prime_index, exponent) pairs
        factors = []
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((i, count))
        if temp > 1:
            # temp is a prime > 53, treat as a single prime factor
            factors.append((nprimes, 1))
        
        # Build all divisors with their prime-exponent bitmasks
        # Each divisor is represented as (value, mask) where mask has bits
        # set for primes that divide it
        divs = [(1, 0)]
        for pi, exp in factors:
            cur_len = len(divs)
            p = primes[pi] if pi < nprimes else temp
            mask_bit = 1 << pi
            factor = 1
            for _ in range(exp):
                factor *= p
                for j in range(cur_len):
                    divs.append((divs[j][0] * factor, divs[j][1] | mask_bit))
        
        divs.sort()
        score = 0
        m = len(divs)
        for i in range(m - 1):
            if divs[i][1] & divs[i+1][1] == 0:
                score += 1
        return score

    def feasible(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
                if prod > max_val:
                    return False
        return True

    def score_exps(exps):
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
        return tau_perp(prod), prod

    # ---- DP-based scoring: process primes in order, track mask state ----
    # State: (prime_idx, mask) -> max score achievable
    # Transition: for each prime i with exponent e_i, we choose which subset of
    # primes {0..i} appear in the "current divisor" (represented by mask)
    # The score increments when consecutive divisors have disjoint masks
    
    def dp_score(exps):
        """
        Compute tau_perp using DP over prime indices.
        State: after processing primes 0..i, what's the best score
        given that the "last divisor" has prime mask = mask.
        """
        # We'll use a dictionary: mask -> best score
        # Start: before any primes, we have divisor 1 with mask 0
        dp = {0: 0}
        
        for i in range(nprimes):
            e = exps[i]
            if e == 0:
                continue
            
            new_dp = {}
            # For each existing state (mask, score), we can either:
            # 1. Not include prime i in the new divisor (mask stays same)
            # 2. Include prime i (mask gets bit i set)
            # But we need to think about how divisors are ordered...
            
            # Actually, the DP needs to track the sequence of divisors.
            # Let's use a different approach: 
            # For a given set of primes with exponents, the divisors are all
            # products p_0^a_0 * p_1^a_1 * ... where 0 <= a_i <= e_i.
            # We want to count adjacent pairs with gcd 1.
            
            # Key insight: two divisors d1, d2 are coprime iff they share no prime.
            # So we need to count adjacent pairs in the sorted list that are coprime.
            
            # For DP, process primes one at a time. When we add prime i with exponent e_i,
            # each existing divisor d can be multiplied by p_i^k for k=0..e_i.
            # The new divisors are interleaved in sorted order.
            
            # This is complex. Let's use a simpler heuristic: 
            # The score is roughly proportional to the number of "transitions"
            # between divisors that don't share primes.
            
            # For now, fall back to exact computation but with memoization
            pass
        
        # Fallback: use exact tau_perp
        prod = 1
        for i, e in enumerate(exps):
            if e:
                prod *= primes[i] ** e
        return tau_perp(prod)

    # ---- Local search with swap moves ----
    def local_search(exps, iters=3000):
        cur_exps = list(exps)
        cur_score, cur_prod = score_exps(cur_exps)
        for _ in range(iters):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            improved = False
            # Standard increase/decrease moves
            for i in range(nprimes):
                # increase
                new_exps = cur_exps[:]
                new_exps[i] += 1
                if feasible(new_exps):
                    s, pr = score_exps(new_exps)
                    if s > cur_score:
                        cur_exps, cur_score, cur_prod = new_exps, s, pr
                        improved = True
                # decrease
                if cur_exps[i] > 0:
                    new_exps = cur_exps[:]
                    new_exps[i] -= 1
                    s, pr = score_exps(new_exps)
                    if s > cur_score:
                        cur_exps, cur_score, cur_prod = new_exps, s, pr
                        improved = True
            # Swap moves: replace prime i (with exp 1) with prime j (exp 0)
            # This keeps the number of distinct primes the same
            for i in range(nprimes):
                if cur_exps[i] == 1:
                    for j in range(nprimes):
                        if cur_exps[j] == 0 and j != i:
                            new_exps = cur_exps[:]
                            new_exps[i] = 0
                            new_exps[j] = 1
                            if feasible(new_exps):
                                s, pr = score_exps(new_exps)
                                if s > cur_score:
                                    cur_exps, cur_score, cur_prod = new_exps, s, pr
                                    improved = True
            if not improved:
                break
        return cur_score, cur_prod, cur_exps

    # ---- Local search over exponent vectors ----
    def local_search(exps, iters=3000):
        cur_exps = list(exps)
        cur_score, cur_prod = score_exps(cur_exps)
        for _ in range(iters):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            improved = False
            for i in range(16):
                # increase
                if feasible(cur_exps[:i] + [cur_exps[i] + 1] + cur_exps[i + 1:]):
                    new = cur_exps[:i] + [cur_exps[i] + 1] + cur_exps[i + 1:]
                    s, pr = score_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score, cur_prod = new, s, pr
                        improved = True
                # decrease
                if cur_exps[i] > 0:
                    new = cur_exps[:i] + [cur_exps[i] - 1] + cur_exps[i + 1:]
                    s, pr = score_exps(new)
                    if s > cur_score:
                        cur_exps, cur_score, cur_prod = new, s, pr
                        improved = True
            if not improved:
                break
        return cur_score, cur_prod, cur_exps

    # ---- Phase 1: square-free subsets ----
    best_score = -1
    best_n = 2
    best_exps = [0] * 16

    # Check contiguous prefix square-free bases
    prod = 1
    for k in range(16):
        if k < 15:
            prod *= primes[k]
        if prod > max_val:
            break
        exps = [1] * (k + 1) + [0] * (15 - k)
        s = tau_perp(prod)
        if s > best_score:
            best_score, best_n, best_exps = s, prod, exps

    # Also check some non-contiguous square-free bases of size 14 and 15
    # by swapping one prime in the prefix
    for k in [14, 15]:
        base_prod = 1
        for i in range(k):
            base_prod *= primes[i]
        if base_prod > max_val:
            continue
        # Try replacing each prime in the prefix with a larger one
        for i in range(k):
            for j in range(k, nprimes):
                new_prod = base_prod // primes[i] * primes[j]
                if new_prod > max_val:
                    continue
                exps = [0] * nprimes
                for idx in range(k):
                    if idx != i:
                        exps[idx] = 1
                exps[j] = 1
                if feasible(exps):
                    s = tau_perp(new_prod)
                    if s > best_score:
                        best_score, best_n, best_exps = s, new_prod, exps

    # ---- Phase 2: refine near-optimal square-free vectors ----
    # Start from the best square-free base found in Phase 1
    if best_score > 0:
        s, pr, _ = local_search(best_exps, iters=3000)
        if s > best_score:
            best_score, best_n, best_exps = s, pr, _

    # Also try from contiguous prefixes of size 13-15
    for k in range(13, 16):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        base = [1] * (k + 1) + [0] * (15 - k)
        if not feasible(base):
            continue
        s, pr, _ = local_search(base, iters=1500)
        if s > best_score:
            best_score, best_n, best_exps = s, pr, _

    # ---- Phase 3: targeted randomized restarts ----
    rng = random.Random(12345)
    for restart in range(30):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        exps = [0] * nprimes
        # Start from a random square-free subset of size 13-15
        target_size = rng.randint(13, 15)
        indices = list(range(nprimes))
        rng.shuffle(indices)
        for i in range(target_size):
            exps[indices[i]] = 1
        if not feasible(exps):
            continue
        s, pr, _ = local_search(exps, iters=1000)
        if s > best_score:
            best_score, best_n, best_exps = s, pr, _

    return best_n
