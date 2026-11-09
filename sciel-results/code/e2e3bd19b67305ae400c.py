"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import isqrt
from collections import defaultdict

# Precompute primes up to a limit to speed up sigma calculation
_PRIME_LIMIT = 200000
_primes = []
_is_prime = [True] * (_PRIME_LIMIT + 1)
_is_prime[0] = False
_is_prime[1] = False
for i in range(2, int(_PRIME_LIMIT**0.5) + 1):
    if _is_prime[i]:
        _is_prime[i*i::_PRIME_LIMIT+1] = [False] * len(_is_prime[i*i::_PRIME_LIMIT+1])
for i in range(2, _PRIME_LIMIT + 1):
    if _is_prime[i]:
        _primes.append(i)

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using precomputed primes."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    for p in _primes:
        if p * p > temp:
            break
        if temp % p == 0:
            p_pow = 1
            while temp % p == 0:
                temp //= p
                p_pow *= p
            result *= (p * p_pow - 1) // (p - 1)
    if temp > 1:
        result *= (temp + 1)
    return result

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    if n <= 1:
        return []
    factors = []
    temp = n
    if temp % 2 == 0:
        cnt = 0
        while temp % 2 == 0:
            temp //= 2
            cnt += 1
        factors.append((2, cnt))
    f = 3
    while f * f <= temp:
        if temp % f == 0:
            cnt = 0
            while temp % f == 0:
                temp //= f
                cnt += 1
            factors.append((f, cnt))
        f += 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _all_divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        p_pow = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_pow)
            p_pow *= p
        divisors = new_divs
    return divisors

def _generate_highly_composite_candidates(deadline):
    """Generate candidate M values that are products of small primes with various exponents."""
    candidates = set()
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    # Generate M = product of p_i^e_i where e_i >= 1 for first few primes
    # Focus on combinations that produce many divisors
    def gen(idx, current, max_val):
        if time.time() > deadline:
            return
        if idx >= len(small_primes):
            if current > 1 and current <= 10**18:
                candidates.add(current)
            return
        p = small_primes[idx]
        val = current
        for e in range(0, 15):
            if val > max_val:
                break
            if val > 1 or e > 0:
                candidates.add(val)
            gen(idx + 1, val, max_val)
            val *= p
            if val > max_val:
                break
    
    # Generate with limited depth to avoid explosion
    gen(0, 1, 10**18)
    
    # Also add some specific highly composite numbers
    known_hc = [
        720720, 1441440, 4324320, 21621600, 75675600, 108108000,
        216216000, 360360000, 720720000, 1441440000, 4324320000,
        8648640000, 25945920000, 51891840000, 155675520000,
        311351040000, 622702080000, 1245404160000, 2490808320000,
        4981616640000, 9963233280000, 19926466560000, 39852933120000,
        79705866240000, 159411732480000, 318823464960000,
    ]
    for hc in known_hc:
        if hc <= 10**18:
            candidates.add(hc)
    
    return candidates

def _find_k_for_M_direct(M, deadline):
    """
    Find all k <= sqrt(M) such that k * sigma(k) = M.
    Enumerate divisors k of M up to sqrt(M), check k * sigma(k) == M.
    """
    sqrtM = isqrt(M)
    results = []
    factors = _factorize(M)
    
    # If M has too many divisors, this could be slow
    num_divs = 1
    for p, e in factors:
        num_divs *= (e + 1)
    
    if num_divs > 100000:
        # Too many divisors, use a different approach: scan k up to some limit
        # But only k that divide M
        # Generate divisors incrementally and check
        divisors = _all_divisors_from_factors(factors)
        for k in divisors:
            if time.time() > deadline:
                break
            if k > sqrtM:
                continue
            if k < 1:
                continue
            s = _sigma(k)
            if k * s == M:
                results.append(k)
    else:
        divisors = _all_divisors_from_factors(factors)
        for k in divisors:
            if k > sqrtM:
                continue
            if k < 1:
                continue
            s = _sigma(k)
            if k * s == M:
                results.append(k)
    
    return results

def _search_collisions_linear(max_k, deadline):
    """
    Search for collisions in k*sigma(k) for k up to max_k.
    Returns dict: M -> list of k values.
    """
    collision_map = defaultdict(list)
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            collision_map[M].append(k)
    
    return collision_map

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline to be safe
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Phase 1: Linear scan for collisions
    search_limit = 5_000_000
    collision_map = defaultdict(list)
    
    for k in range(1, search_limit + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            collision_map[M].append(k)
    
    # Find the M with the most valid solutions
    for M, ks_list in collision_map.items():
        if len(ks_list) < best_count:
            continue
        sqrtM = isqrt(M)
        valid_ks = [k for k in ks_list if k <= sqrtM]
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_ks = valid_ks
    
    # Phase 2: Enumerate highly composite M candidates
    if time.time() < deadline - 10:
        candidates = _generate_highly_composite_candidates(deadline)
        
        # Sort candidates by number of divisors (heuristic: more divisors = more potential k)
        # Actually, let's just try them and see which give many solutions
        # Limit the number we try
        candidate_list = sorted(candidates, key=lambda m: m)
        
        for M in candidate_list:
            if time.time() > deadline - 5:
                break
            if M > 10**18:
                continue
            if M < 1:
                continue
            
            # Quick check: how many divisors does M have?
            factors = _factorize(M)
            num_divs = 1
            for p, e in factors:
                num_divs *= (e + 1)
            
            # If too many divisors, skip (would be too slow to check all)
            if num_divs > 50000:
                continue
            
            # Find k values for this M
            ks = _find_k_for_M_direct(M, deadline)
            sqrtM = isqrt(M)
            valid_ks = [k for k in ks if k <= sqrtM]
            
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M
                best_ks = valid_ks
    
    # Phase 3: If we have time, extend the linear search
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        extended_limit = search_limit + int(remaining_time * 500000)  # ~500k per second
        for k in range(search_limit + 1, min(extended_limit, 30_000_000) + 1):
            if time.time() > deadline - 2:
                break
            s = _sigma(k)
            M = k * s
            if M <= 10**18:
                collision_map[M].append(k)
        
        # Re-evaluate
        for M, ks_list in collision_map.items():
            if len(ks_list) < best_count:
                continue
            sqrtM = isqrt(M)
            valid_ks = [k for k in ks_list if k <= sqrtM]
            if len(valid_ks) > best_count:
                best_count = len(valid_ks)
                best_M = M
                best_ks = valid_ks
    
    # Verify the best result
    sqrtM = isqrt(best_M)
    verified_ks = []
    for k in best_ks:
        if k <= sqrtM and k * _sigma(k) == best_M:
            verified_ks.append(k)
    
    # Ensure distinct
    verified_ks = list(set(verified_ks))
    
    if len(verified_ks) == 0:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
# EVOLVE-BLOCK-END
