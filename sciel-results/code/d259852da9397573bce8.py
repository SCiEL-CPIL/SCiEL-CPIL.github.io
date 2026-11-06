"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Structural approach: bounded search over prime exponent vectors with
bitset-accelerated coprime detection and expanded prime substitution search.
"""
import math
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
def _tau_perp(n):
    """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort()
    count = 0
    for j in range(len(divisors) - 1):
        if math.gcd(divisors[j], divisors[j+1]) == 1:
            count += 1
    return count

def _tau_perp_bitset(primes, exponents):
    """
    Compute tau_perp using bitset representation of prime supports.
    Each divisor is represented by a bitmask of which primes divide it.
    Two divisors are coprime iff their bitmasks are disjoint (AND == 0).
    """
    num_primes = len(primes)
    # Generate all divisors as (value, bitmask) pairs
    divs = [(1, 0)]
    for idx, (p, e) in enumerate(zip(primes, exponents)):
        new_divs = []
        p_powers = [1]
        for k in range(1, e + 1):
            p_powers.append(p_powers[-1] * p)
        for val, mask in divs:
            for k in range(e + 1):
                new_mask = mask | ((1 << idx) if k > 0 else 0)
                new_divs.append((val * p_powers[k], new_mask))
        divs = new_divs
    
    # Sort by value
    divs.sort(key=lambda x: x[0])
    
    count = 0
    for j in range(len(divs) - 1):
        if divs[j][1] & divs[j+1][1] == 0:
            count += 1
    return count

def _is_prime(n):
    """Miller-Rabin primality test for n up to 10^18."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic witnesses for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def _primes_up_to(limit):
    """Sieve of Eratosthenes for primes up to limit."""
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Generate primes up to ~200 for substitution candidates
    max_prime_search = 200
    all_primes = _primes_up_to(max_prime_search)
    # Ensure we have at least 30 primes
    if len(all_primes) < 30:
        # Extend if needed
        max_prime_search = 300
        all_primes = _primes_up_to(max_prime_search)
    
    base_primes = all_primes[:25]
    
    best_n = 1
    best_score = 0
    
    # Phase 1: Square-free with 15 primes (the primorial)
    p15 = base_primes[:15]
    n15 = 1
    for p in p15:
        n15 *= p
    if n15 <= 10**18:
        score = _tau_perp_bitset(p15, [1]*15)
        if score > best_score:
            best_score = score
            best_n = n15
    
    # Phase 2: Try swapping the largest 1-4 primes with larger ones
    # Keep the product <= 10^18
    # First 14 primes product: ~1.3e13, so 15th prime can be up to ~7.7e4
    # First 13 primes product: ~3e11, so 14th*15th product can be up to ~3.3e6
    # First 12 primes product: ~7.7e9, so 13th*14th*15th product can be up to ~1.3e8
    
    # Precompute prefix products
    prefix_prod = [1]
    for p in base_primes[:15]:
        prefix_prod.append(prefix_prod[-1] * p)
    
    # Phase 2a: Replace last 1 prime (15 total primes)
    fixed14 = base_primes[:14]
    fixed14_prod = prefix_prod[14]
    max_p15 = 10**18 // fixed14_prod
    # Try all primes from index 13 to max_p15
    for p in all_primes:
        if p > max_p15:
            break
        if p in fixed14:
            continue
        primes_list = fixed14 + [p]
        n = fixed14_prod * p
        score = _tau_perp_bitset(primes_list, [1]*15)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Phase 2b: Replace last 2 primes (15 total primes)
    if time.time() > deadline:
        return best_n
    fixed13 = base_primes[:13]
    fixed13_prod = prefix_prod[13]
    max_prod2 = 10**18 // fixed13_prod
    # Try pairs of primes whose product <= max_prod2
    candidate_primes_2b = [p for p in all_primes if p not in set(fixed13)]
    for i in range(len(candidate_primes_2b)):
        p1 = candidate_primes_2b[i]
        if p1 > max_prod2:
            break
        max_p2 = max_prod2 // p1
        for j in range(i + 1, len(candidate_primes_2b)):
            p2 = candidate_primes_2b[j]
            if p2 > max_p2:
                break
            primes_list = fixed13 + [p1, p2]
            n = fixed13_prod * p1 * p2
            score = _tau_perp_bitset(primes_list, [1]*15)
            if score > best_score:
                best_score = score
                best_n = n
    
    # Phase 2c: Replace last 3 primes (15 total primes)
    if time.time() > deadline:
        return best_n
    fixed12 = base_primes[:12]
    fixed12_prod = prefix_prod[12]
    max_prod3 = 10**18 // fixed12_prod
    candidate_primes_2c = [p for p in all_primes if p not in set(fixed12)]
    # Only try if feasible
    if candidate_primes_2c[0] * candidate_primes_2c[1] * candidate_primes_2c[2] <= max_prod3:
        for i in range(len(candidate_primes_2c)):
            p1 = candidate_primes_2c[i]
            if p1 > max_prod3:
                break
            max_prod2_rem = max_prod3 // p1
            for j in range(i + 1, len(candidate_primes_2c)):
                p2 = candidate_primes_2c[j]
                if p2 > max_prod2_rem:
                    break
                max_p3 = max_prod2_rem // p2
                for k in range(j + 1, len(candidate_primes_2c)):
                    p3 = candidate_primes_2c[k]
                    if p3 > max_p3:
                        break
                    primes_list = fixed12 + [p1, p2, p3]
                    n = fixed12_prod * p1 * p2 * p3
                    score = _tau_perp_bitset(primes_list, [1]*15)
                    if score > best_score:
                        best_score = score
                        best_n = n
    
    # Phase 3: Try 14 primes with one prime squared
    if time.time() > deadline:
        return best_n
    
    # For each choice of which prime to square, try different sets of 13 other primes
    for sq_idx in range(6):  # Square primes at index 0-5
        if time.time() > deadline:
            break
        sq_prime = base_primes[sq_idx]
        sq_prod = sq_prime ** 2
        
        # We need 13 other distinct primes, none equal to sq_prime
        # Try greedy: smallest available primes first
        other_primes = []
        prod = sq_prod
        for p in base_primes[:20]:
            if p == sq_prime:
                continue
            if prod * p > 10**18:
                break
            other_primes.append(p)
            prod *= p
            if len(other_primes) == 13:
                break
        
        if len(other_primes) == 13:
            primes_list = [sq_prime] + other_primes
            exponents = [2] + [1]*13
            n = prod
            score = _tau_perp_bitset(primes_list, exponents)
            if score > best_score:
                best_score = score
                best_n = n
        
        # Also try replacing the largest 1-2 of the 13 other primes with larger ones
        if len(other_primes) == 13 and time.time() < deadline:
            fixed_other = other_primes[:11]
            fixed_other_prod = sq_prod
            for p in fixed_other:
                fixed_other_prod *= p
            max_last2 = 10**18 // fixed_other_prod
            cand = [p for p in all_primes if p != sq_prime and p not in set(fixed_other)]
            for i in range(len(cand)):
                p1 = cand[i]
                if p1 > max_last2:
                    break
                max_p2 = max_last2 // p1
                for j in range(i + 1, len(cand)):
                    p2 = cand[j]
                    if p2 > max_p2:
                        break
                    primes_list = [sq_prime] + fixed_other + [p1, p2]
                    exponents = [2] + [1]*13
                    n = fixed_other_prod * p1 * p2
                    score = _tau_perp_bitset(primes_list, exponents)
                    if score > best_score:
                        best_score = score
                        best_n = n
    
    # Phase 4: Try 13 primes with two primes squared
    if time.time() > deadline:
        return best_n
    
    for sq1_idx, sq2_idx in itertools.combinations(range(5), 2):
        if time.time() > deadline:
            break
        sq1 = base_primes[sq1_idx]
        sq2 = base_primes[sq2_idx]
        prod = sq1 ** 2 * sq2 ** 2
        
        # Add 11 more distinct primes
        other_primes = []
        for i in range(20):
            if i == sq1_idx or i == sq2_idx:
                continue
            p = base_primes[i]
            if prod * p > 10**18:
                break
            other_primes.append(p)
            prod *= p
            if len(other_primes) == 11:
                break
        
        if len(other_primes) == 11:
            primes_list = [sq1, sq2] + other_primes
            exponents = [2, 2] + [1]*11
            n = prod
            score = _tau_perp_bitset(primes_list, exponents)
            if score > best_score:
                best_score = score
                best_n = n
    
    # Phase 5: Try 14 primes with one prime cubed (p^3)
    if time.time() > deadline:
        return best_n
    
    for cube_idx in range(4):  # Cube primes at index 0-3
        if time.time() > deadline:
            break
        cube_prime = base_primes[cube_idx]
        cube_prod = cube_prime ** 3
        
        other_primes = []
        prod = cube_prod
        for p in base_primes[:20]:
            if p == cube_prime:
                continue
            if prod * p > 10**18:
                break
            other_primes.append(p)
            prod *= p
            if len(other_primes) == 13:
                break
        
        if len(other_primes) == 13:
            primes_list = [cube_prime] + other_primes
            exponents = [3] + [1]*13
            n = prod
            score = _tau_perp_bitset(primes_list, exponents)
            if score > best_score:
                best_score = score
                best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
