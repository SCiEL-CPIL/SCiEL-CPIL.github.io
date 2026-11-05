import math
import time
import heapq
from functools import reduce

LIMIT = 10**18

def _score_from_factors(primes, exps):
    """
    Compute tau_perp(n) where n = prod(primes[i]^exps[i]).
    Generate divisors in sorted order using a min-heap, then count coprime adjacent pairs.
    """
    k = len(primes)
    if k == 0:
        return 0
    if k == 1:
        # Only one prime: all divisors are powers of p, no two are coprime (except 1 and p^0=1... wait)
        # Divisors: 1, p, p^2, ..., p^e
        # gcd(1, p) = 1, gcd(p, p^2) = p > 1, etc.
        # So only the first pair (1, p) is coprime if e >= 1
        return 1 if exps[0] >= 1 else 0
    
    # Generate all divisors in sorted order
    # Use the standard technique: maintain a heap of (value, index_tuple)
    # Start with 1, then for each divisor d, we can generate d * p_i for each i where exp[i] < max
    
    # For efficiency with potentially many divisors, use the "sorted generation" approach
    # n = p1^e1 * p2^e2 * ... * pk^ek
    # Divisors can be generated as: start with 1, then repeatedly multiply by primes
    
    # Precompute prime powers
    prime_powers = []
    for i, p in enumerate(primes):
        pp = [1]
        for e in range(1, exps[i] + 1):
            pp.append(pp[-1] * p)
        prime_powers.append(pp)
    
    # Total number of divisors
    total_divs = 1
    for e in exps:
        total_divs *= (e + 1)
    
    if total_divs > 5_000_000:
        # Too many divisors to enumerate; use an approximation or skip
        # This shouldn't happen for our candidates
        return 0
    
    # Generate divisors using a min-heap
    # Each element: (value, (e0, e1, ..., ek-1))
    # Start with (1, (0, 0, ..., 0))
    # From (v, e), we can go to (v * p_i, e with e_i+1) if e_i < exps[i]
    
    # Use a different approach: generate all divisors, sort them
    # For up to ~100K divisors this is fine
    divisors = [1]
    for i in range(k):
        p = primes[i]
        e = exps[i]
        new_divs = []
        for d in divisors:
            val = d
            for j in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    divisors.sort()
    
    # Count coprime adjacent pairs
    score = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            score += 1
    return score


def _score_fast_tuple(primes, exps):
    """
    Faster scoring using tuple representation.
    Two divisors (as exponent tuples) are coprime iff they have no common prime with positive exponent.
    """
    k = len(primes)
    if k == 0:
        return 0
    
    # Generate all divisor exponent tuples
    from itertools import product
    ranges = [range(e + 1) for e in exps]
    div_tuples = list(product(*ranges))
    
    # Sort by integer value
    def tuple_val(t):
        v = 1
        for i in range(k):
            if t[i] > 0:
                v *= primes[i] ** t[i]
        return v
    
    div_tuples.sort(key=tuple_val)
    
    # Count adjacent coprime pairs
    score = 0
    for i in range(len(div_tuples) - 1):
        a = div_tuples[i]
        b = div_tuples[i + 1]
        coprime = True
        for j in range(k):
            if a[j] > 0 and b[j] > 0:
                coprime = False
                break
        if coprime:
            score += 1
    return score


def _eval_candidate(primes, exps):
    """Evaluate a candidate, return (score, n) or (0, 0) if invalid."""
    n = 1
    for p, e in zip(primes, exps):
        if e > 0:
            n *= p ** e
        if n > LIMIT:
            return (0, 0)
    if n < 2:
        return (0, 0)
    score = _score_fast_tuple(primes, exps)
    return (score, n)


def _search_exponents(primes, max_exps, time_limit, start_time):
    """
    Search over exponent assignments for given primes.
    Uses a greedy + local search approach.
    """
    deadline = start_time + time_limit
    best_score = 0
    best_exps = None
    best_n = 0
    
    k = len(primes)
    
    # Start with all 1s
    exps = [1] * k
    n = 1
    for i, p in enumerate(primes):
        n *= p
    if n > LIMIT:
        # Can't even fit all with exponent 1
        return (0, 0)
    
    score, _ = _eval_candidate(primes, exps)
    if score > best_score:
        best_score = score
        best_exps = list(exps)
        best_n = n
    
    # Local search: try changing one exponent at a time
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in range(k):
            # Try increasing
            if exps[i] < max_exps[i]:
                old_exp = exps[i]
                exps[i] = old_exp + 1
                s, n_new = _eval_candidate(primes, exps)
                if s > best_score:
                    best_score = s
                    best_exps = list(exps)
                    best_n = n_new
                    improved = True
                else:
                    exps[i] = old_exp
            
            # Try decreasing (if > 0)
            if exps[i] > 1:
                old_exp = exps[i]
                exps[i] = old_exp - 1
                s, n_new = _eval_candidate(primes, exps)
                if s > best_score:
                    best_score = s
                    best_exps = list(exps)
                    best_n = n_new
                    improved = True
                else:
                    exps[i] = old_exp
    
    return (best_score, best_n)


def _greedy_build(primes_pool, time_limit, start_time):
    """
    Greedily build n by adding primes one at a time, choosing the best exponent.
    """
    deadline = start_time + time_limit
    best_score = 0
    best_n = 2
    
    # Try different numbers of primes
    for num_primes in range(2, len(primes_pool) + 1):
        if time.time() > deadline:
            break
        
        chosen = primes_pool[:num_primes]
        
        # Max possible exponent for each prime
        max_exps = []
        for p in chosen:
            max_e = 0
            val = 1
            while val * p <= LIMIT:
                val *= p
                max_e += 1
            max_exps.append(max_e)
        
        # Start with all 1s
        exps = [1] * num_primes
        n = 1
        for i, p in enumerate(chosen):
            n *= p
        if n > LIMIT:
            continue
        
        score, _ = _eval_candidate(chosen, exps)
        if score > best_score:
            best_score = score
            best_n = n
        
        # Local search
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            for i in range(num_primes):
                # Try increasing
                if exps[i] < max_exps[i]:
                    old_exp = exps[i]
                    exps[i] = old_exp + 1
                    s, n_new = _eval_candidate(chosen, exps)
                    if s > best_score:
                        best_score = s
                        best_n = n_new
                        improved = True
                    else:
                        exps[i] = old_exp
                
                # Try decreasing
                if exps[i] > 1:
                    old_exp = exps[i]
                    exps[i] = old_exp - 1
                    s, n_new = _eval_candidate(chosen, exps)
                    if s > best_score:
                        best_score = s
                        best_n = n_new
                        improved = True
                    else:
                        exps[i] = old_exp
        
        # Also try biased starts: high exponent on first few primes
        for bias in range(2, 6):
            if time.time() > deadline:
                break
            exps2 = [1] * num_primes
            # Give extra to first prime
            remaining = LIMIT // (chosen[0] ** bias)
            exps2[0] = bias
            for i in range(1, num_primes):
                if remaining * chosen[i] <= LIMIT:
                    remaining *= chosen[i]
                else:
                    break
            if exps2[0] > 1:
                s2, n2 = _eval_candidate(chosen, exps2)
                if s2 > best_score:
                    best_score = s2
                    best_n = n2
    
    return best_n, best_score


def _try_structured_candidates():
    """Try a set of carefully structured candidates."""
    best_score = 0
    best_n = 2
    
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    # Product of first k primes
    for k in range(2, 20):
        if k > len(primes_list):
            break
        n = 1
        for p in primes_list[:k]:
            n *= p
            if n > LIMIT:
                break
        if n > LIMIT:
            break
        s = _score_fast_tuple(primes_list[:k], [1] * k)
        if s > best_score:
            best_score = s
            best_n = n
    
    # Various exponent patterns
    patterns = [
        # (num_primes, exponents)
        (14, [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [7, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [9, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [10, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [11, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [12, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [5, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [6, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [6, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [8, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [8, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [9, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [10, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [10, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [5, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [5, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [6, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [7, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [7, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [8, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [8, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [5, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [5, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [6, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [6, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [7, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (14, [7, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [7, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [9, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [10, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [5, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [6, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [6, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [8, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [8, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [5, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [5, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [6, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [7, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (15, [7, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [7, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [5, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [6, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [6, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [5, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [5, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [6, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (16, [6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (17, [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (17, [6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (17, [7, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (17, [5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (17, [5, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (17, [6, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (17, [5, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (17, [5, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (18, [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (18, [6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (18, [5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (18, [5, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (19, [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (19, [6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (19, [5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (20, [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (20, [5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    ]
    
    for num_primes, exps in patterns:
        if num_primes > len(primes_list):
            continue
        chosen = primes_list[:num_primes]
        if len(exps) != num_primes:
            continue
        n = 1
        valid = True
        for i, p in enumerate(chosen):
            if exps[i] > 0:
                n *= p ** exps[i]
            if n > LIMIT:
                valid = False
                break
        if not valid or n < 2:
            continue
        s = _score_fast_tuple(chosen, exps)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # Phase 1: Try structured candidates
    best_n, best_score = _try_structured_candidates()
    
    if time.time() > deadline:
        return best_n if best_n >= 2 else 2
    
    # Phase 2: Greedy search with local refinement
    primes_pool = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    remaining_time = deadline - time.time()
    greedy_n, greedy_score = _greedy_build(primes_pool, remaining_time * 0.5, time.time())
    if greedy_score > best_score:
        best_score = greedy_score
        best_n = greedy_n
    
    if time.time() > deadline:
        return best_n if best_n >= 2 else 2
    
    # Phase 3: More targeted local search from best candidate
    if time.time() < deadline - 5:
        # Factor best_n to get primes and exponents
        n = best_n
        factors = {}
        d = 2
        x = n
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1 if d == 2 else 2
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        
        primes = sorted(factors.keys())
        exps = [factors[p] for p in primes]
        
        # Local search
        improved = True
        while improved and (time.time() < deadline - 2):
            improved = False
            for i in range(len(primes)):
                # Try increasing
                old_exp = exps[i]
                new_exp = old_exp + 1
                new_n = best_n * primes[i]
                if new_n <= LIMIT:
                    exps[i] = new_exp
                    s = _score_fast_tuple(primes, exps)
                    if s > best_score:
                        best_score = s
                        best_n = new_n
                        improved = True
                    else:
                        exps[i] = old_exp
                
                # Try decreasing
                if old_exp > 1:
                    new_exp = old_exp - 1
                    new_n = best_n // primes[i]
                    exps[i] = new_exp
                    s = _score_fast_tuple(primes, exps)
                    if s > best_score:
                        best_score = s
                        best_n = new_n
                        improved = True
                    else:
                        exps[i] = old_exp
    
    return best_n if best_n >= 2 else 2
