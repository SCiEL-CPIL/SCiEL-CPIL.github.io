import math
from functools import lru_cache

def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent 
    coprime pairs in its sorted divisor list.
    
    Approach: We build n = p1^a1 * p2^a2 * ... * pk^ak where p_i are the first k primes.
    The sorted divisors of n correspond to all tuples (e1, ..., ek) with 0 <= ei <= ai,
    ordered by the resulting product. We want to maximize the number of adjacent pairs
    (in sorted order) that are coprime.
    
    Two divisors d1 and d2 are coprime iff for every prime p_i, at least one of d1 or d2
    has exponent 0 for p_i.
    
    For a primorial (all exponents = 1), the divisors are all square-free products of subsets
    of {p1, ..., pk}. Two such divisors are coprime iff they share no common prime, i.e., their
    index sets are disjoint.
    
    The key insight: for a primorial, the number of coprime adjacent pairs in the sorted divisor
    sequence can be computed. We want to maximize this over k such that p1*...*pk <= 10^18.
    
    However, we can also consider non-primorial numbers. Let's think about when exponents > 1 help.
    If we have p1^2, the divisors include 1, p1, p1^2, ... The adjacency structure changes.
    
    For the primorial case, let's compute the score for each k and pick the best one.
    
    Actually, let's directly compute tau_perp for the primorial of order k by generating all divisors,
    sorting them, and counting adjacent coprime pairs. This is feasible for k up to ~15 since 2^15 = 32768.
    
    The largest primorial <= 10^18:
    2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = let's compute.
    
    2*3=6, *5=30, *7=210, *11=2310, *13=30030, *17=510510, *19=9699690,
    *23=223092870, *29=6469693230, *31=200560490130, *37=7420738134810,
    *41=304250263527210, *43=13082761331670030, *47=614889782588491410,
    *53=32589158477190044730 > 10^18? 10^18 = 1000000000000000000
    614889782588491410 * 53 = 32589158477190044730 which is > 10^18.
    So k=15 gives primorial = 614889782588491410 <= 10^18.
    k=16 would exceed.
    
    So we have primorials for k=1..15. Let's compute tau_perp for each and pick the max.
    
    But wait - can we do better with non-primorial? E.g., use fewer primes but with higher exponents?
    Let's also try some hybrid constructions and pick the best overall.
    
    For efficiency, I'll compute the score for primorials k=1..15, and also try a few other candidates.
    """
    
    def get_primes(count):
        """Generate first `count` primes."""
        primes = []
        candidate = 2
        while len(primes) < count:
            # trial division
            is_p = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(candidate)
            candidate += 1 if len(primes) < 2 else 2
        return primes
    
    def compute_tau_perp(primes, exponents):
        """
        Given a list of primes and corresponding exponents, compute the number of
        adjacent coprime pairs in the sorted divisor list.
        """
        k = len(primes)
        # Generate all divisors as (product, mask_of_primes_with_positive_exp)
        # Actually for coprimality we need: for each prime, whether the divisor has exp > 0.
        # Two divisors are coprime iff for all i, not (exp_i(d1) > 0 and exp_i(d2) > 0).
        
        divisors = []  # list of (value, bitmask) where bitmask has bit i set if exp_i > 0
        
        def gen(idx, val, mask):
            if idx == k:
                divisors.append((val, mask))
                return
            p = primes[idx]
            e = exponents[idx]
            for exp in range(e + 1):
                new_mask = mask | (1 << idx) if exp > 0 else mask
                gen(idx + 1, val, new_mask)
                val *= p
                if val > 10**18 + 1:
                    break
        
        gen(0, 1, 0)
        
        # Sort by value
        divisors.sort(key=lambda x: x[0])
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(divisors) - 1):
            mask1 = divisors[i][1]
            mask2 = divisors[i+1][1]
            if (mask1 & mask2) == 0:
                score += 1
        return score
    
    # Generate first 16 primes
    primes = get_primes(16)
    
    # Try primorials (all exponents = 1) for k = 1..15
    best_score = 0
    best_n = 2
    
    for k in range(1, 16):
        # Check if product of first k primes <= 10^18
        prod = 1
        for i in range(k):
            prod *= primes[i]
        if prod > 10**18:
            break
        exponents = [1] * k
        score = compute_tau_perp(primes[:k], exponents)
        if score > best_score:
            best_score = score
            best_n = prod
    
    # Also try some non-primorial candidates to see if they beat the best primorial
    # e.g., 2^a * 3^b * 5^c * ... with small number of primes but higher exponents
    # This is expensive to search exhaustively, so let's try a few heuristics.
    
    # Try: use first m primes, with exponent 2 for some of them
    # For m up to 8 or so, we can try combinations
    for m in range(2, 9):
        base_primes = primes[:m]
        # Try all exponent combinations where each exponent is 1 or 2
        # and product <= 10^18
        # This is 2^m combinations, at most 256 for m=8
        for mask in range(2**m):
            exponents = []
            prod = 1
            valid = True
            for i in range(m):
                exp = 1 + (1 if (mask >> i) & 1 else 0)
                exponents.append(exp)
                for _ in range(exp):
                    prod *= base_primes[i]
                    if prod > 10**18:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            score = compute_tau_perp(base_primes, exponents)
            if score > best_score:
                best_score = score
                best_n = prod
    
    # Try: first m primes with exponent up to 3 for the first few
    for m in range(2, 7):
        base_primes = primes[:m]
        # Try exponents from {1, 2, 3} for first 3 primes, 1 for rest
        for e0 in [1, 2, 3]:
            for e1 in [1, 2, 3]:
                if m >= 3:
                    for e2 in [1, 2, 3]:
                        exponents = [e0, e1, e2] + [1] * (m - 3)
                        prod = 1
                        valid = True
                        for i in range(m):
                            prod *= base_primes[i] ** exponents[i]
                            if prod > 10**18:
                                valid = False
                                break
                        if not valid:
                            continue
                        score = compute_tau_perp(base_primes, exponents)
                        if score > best_score:
                            best_score = score
                            best_n = prod
                else:
                    exponents = [e0, e1]
                    prod = 1
                    valid = True
                    for i in range(m):
                        prod *= base_primes[i] ** exponents[i]
                        if prod > 10**18:
                            valid = False
                            break
                    if not valid:
                        continue
                    score = compute_tau_perp(base_primes, exponents)
                    if score > best_score:
                        best_score = score
                        best_n = prod
    
    return best_n
