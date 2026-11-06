"""Constructs an integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Strategy: Search over exponent combinations of small primes, evaluating
tau_perp for each candidate. Uses a deadline guard to stay within limits.
"""

# EVOLVE-BLOCK-START
import math
import time
from itertools import product

def _tau_perp(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    if n < 2:
        return 0
    # Generate all divisors
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    count = 0
    for j in range(len(divs) - 1):
        if math.gcd(divs[j], divs[j + 1]) == 1:
            count += 1
    return count

def _generate_divisors_from_factors(primes, exponents):
    """Generate all divisors from prime factorization."""
    divs = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divs:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp_fast(primes, exponents):
    """Compute tau_perp from prime factorization directly."""
    divs = _generate_divisors_from_factors(primes, exponents)
    count = 0
    for j in range(len(divs) - 1):
        if math.gcd(divs[j], divs[j + 1]) == 1:
            count += 1
    return count

def construct():
    deadline = time.time() + 120  # 120s deadline with margin
    
    # Base primes to consider
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_score = 0
    best_n = 1
    
    # First try the primorial (known good baseline)
    n_primorial = 1
    for p in base_primes:
        n_primorial *= p
    score = _tau_perp(n_primorial)
    best_score = score
    best_n = n_primorial
    
    if time.time() > deadline - 30:
        return best_n
    
    # Search over exponent combinations for subsets of primes
    # Try different prime subsets and exponent patterns
    
    # Strategy: for each subset of primes (up to size ~10), try various exponent patterns
    # Focus on small primes with higher exponents
    
    # Try: 2^a * 3^b * 5^c * 7^d * ... with decreasing exponents
    max_a = 30  # 2^30 ~ 1e9
    max_b = 20
    max_c = 15
    max_d = 12
    
    # Try systematic search over first few primes with varying exponents
    # n = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * ...
    
    # Phase 1: Search over 2^a * 3^b * 5^c * 7^d
    for a in range(1, min(max_a, 30)):
        if time.time() > deadline - 20:
            break
        n_a = 2 ** a
        if n_a > 10**18:
            break
        for b in range(0, min(max_b, 20)):
            n_ab = n_a * (3 ** b)
            if n_ab > 10**18:
                break
            for c in range(0, min(max_c, 15)):
                n_abc = n_ab * (5 ** c)
                if n_abc > 10**18:
                    break
                for d in range(0, min(max_d, 12)):
                    n_abcd = n_abc * (7 ** d)
                    if n_abcd > 10**18:
                        break
                    # Try adding more primes
                    for e in range(0, 6):
                        n_abcde = n_abcd * (11 ** e)
                        if n_abcde > 10**18:
                            break
                        for f in range(0, 5):
                            n_abcdef = n_abcde * (13 ** f)
                            if n_abcdef > 10**18:
                                break
                            # Evaluate
                            primes_used = [2, 3, 5, 7, 11, 13]
                            exps_used = [a, b, c, d, e, f]
                            # Filter out zero exponents
                            actual_primes = []
                            actual_exps = []
                            for p, ex in zip(primes_used, exps_used):
                                if ex > 0:
                                    actual_primes.append(p)
                                    actual_exps.append(ex)
                            if not actual_primes:
                                continue
                            sc = _tau_perp_fast(actual_primes, actual_exps)
                            if sc > best_score:
                                best_score = sc
                                best_n = n_abcdef
                            if time.time() > deadline - 15:
                                break
                        if time.time() > deadline - 15:
                            break
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break
            if time.time() > deadline - 15:
                break
        if time.time() > deadline - 15:
            break
    
    # Phase 2: Try more primes with smaller exponents (square-free-ish)
    if time.time() < deadline - 10:
        # Try all subsets of primes with exponent 1 (square-free)
        # and some with exponent 2
        num_primes = len(base_primes)
        # Try combinations of 8-15 primes
        from itertools import combinations
        for subset_size in range(8, num_primes + 1):
            if time.time() > deadline - 10:
                break
            for combo in combinations(range(num_primes), subset_size):
                if time.time() > deadline - 10:
                    break
                n_val = 1
                primes_used = []
                for idx in combo:
                    n_val *= base_primes[idx]
                    primes_used.append(base_primes[idx])
                    if n_val > 10**18:
                        break
                if n_val > 10**18:
                    continue
                sc = _tau_perp_fast(primes_used, [1] * len(primes_used))
                if sc > best_score:
                    best_score = sc
                    best_n = n_val
    
    # Phase 3: Mixed - some primes with exponent 2
    if time.time() < deadline - 5:
        # Try: 2^2 * 3^2 * 5 * 7 * 11 * ... patterns
        patterns = [
            [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        for pat in patterns:
            if time.time() > deadline - 5:
                break
            n_val = 1
            primes_used = []
            exps_used = []
            valid = True
            for i, ex in enumerate(pat):
                if i >= len(base_primes):
                    break
                if ex > 0:
                    n_val *= base_primes[i] ** ex
                    primes_used.append(base_primes[i])
                    exps_used.append(ex)
                    if n_val > 10**18:
                        valid = False
                        break
            if not valid or not primes_used:
                continue
            sc = _tau_perp_fast(primes_used, exps_used)
            if sc > best_score:
                best_score = sc
                best_n = n_val
    
    return best_n
# EVOLVE-BLOCK-END
