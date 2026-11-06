"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt
from functools import lru_cache

def _factorize(n):
    """Return prime factorization of n as dict {prime: exponent}."""
    if n <= 1:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _divisors_from_factorization(factors):
    """Generate all divisors from prime factorization."""
    if not factors:
        return [1]
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        for d in divisors:
            for e in range(exponents[i] + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    return sorted(divisors)

def _get_factor_pairs(n):
    """Return all (u, v) pairs with u*v = n, u <= v."""
    if n <= 1:
        return [(1, n)] if n == 1 else []
    factors = _factorize(n)
    divs = _divisors_from_factorization(factors)
    pairs = []
    for u in divs:
        if u * u > n:
            break
        if n % u == 0:
            v = n // u
            pairs.append((u, v))
    return pairs

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    target = Fraction(5, 31)
    
    # Known solution: [8, 31, 248], max = 248
    best_solution = [8, 31, 248]
    best_max = 248
    
    def try_solution(denoms):
        nonlocal best_solution, best_max
        if not denoms:
            return
        if len(set(denoms)) != len(denoms):
            return
        if any(d < 1 or d > 10000000 for d in denoms):
            return
        s = sum(Fraction(1, d) for d in denoms)
        if s != target:
            return
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_solution = sorted(denoms)
    
    # Strategy: Binary search on C (max denominator)
    # For each C, check if 5/31 can be written as sum of distinct unit fractions with max denom C
    # Use algebraic solving for the last 1-2 variables to make this fast
    
    def can_solve(rem, min_d, max_d, terms_left, used):
        """
        Check if rem can be written as sum of terms_left distinct unit fractions
        with denominators in [min_d, max_d], none in used.
        Returns list of denominators or None.
        Uses algebraic solving for the last 1-2 variables.
        """
        if rem <= 0:
            return None
        if terms_left == 0:
            return None
        
        p = rem.numerator
        q = rem.denominator
        
        # If rem is a unit fraction
        if q % p == 0:
            d = q // p
            if min_d <= d <= max_d and d not in used:
                return [d]
        
        # Bounds for next denominator
        # 1/d < rem => d > q/p
        # terms_left * (1/d) >= rem => d <= terms_left * q / p
        min_next = max(min_d, q // p + 1)
        max_next = min(max_d, terms_left * q // p)
        
        if min_next > max_next:
            return None
        
        # If only 1 term left
        if terms_left == 1:
            for d in range(min_next, max_next + 1):
                if d in used:
                    continue
                if rem == Fraction(1, d):
                    return [d]
            return None
        
        # If 2 terms left, use algebraic solving
        if terms_left == 2:
            # 1/a + 1/b = rem = p/q
            # (a+b)/(ab) = p/q
            # q(a+b) = p*ab
            # p*ab - q*a - q*b = 0
            # (pa - q)(pb - q) = q^2
            # Let u = pa - q, v = pb - q, then u*v = q^2
            # a = (u+q)/p, b = (v+q)/p
            q2 = q * q
            pairs = _get_factor_pairs(q2)
            for u, v in pairs:
                if time.time() > deadline:
                    return None
                if (u + q) % p != 0:
                    continue
                a = (u + q) // p
                if a < min_next or a > max_next or a in used:
                    continue
                if (v + q) % p != 0:
                    continue
                b = (v + q) // p
                if b <= a or b < min_next or b > max_next or b in used:
                    continue
                # Verify
                if Fraction(1, a) + Fraction(1, b) == rem:
                    return [a, b]
            return None
        
        # For 3+ terms, try each possible next denominator
        for d in range(min_next, max_next + 1):
            if time.time() > deadline:
                return None
            if d in used:
                continue
            new_rem = rem - Fraction(1, d)
            if new_rem <= 0:
                break
            # Pruning: check if remaining can be achieved
            # Max possible sum with (terms_left-1) terms starting from d+1
            # is roughly (terms_left-1)/(d+1)
            if (terms_left - 1) / (d + 1) < new_rem:
                break
            result = can_solve(new_rem, d + 1, max_d, terms_left - 1, used | {d})
            if result is not None:
                return [d] + result
        return None
    
    # Binary search on C
    lo, hi = 7, 248  # We know 248 works, search for smaller
    
    # First, verify that 248 works
    # [8, 31, 248]: 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31 ✓
    
    # Try to find the minimum C
    # We'll do a linear search from low to high, but with smart pruning
    # For small C, the search space is small enough to be fast
    
    # Try C from 7 up to best_max - 1
    for C in range(7, best_max):
        if time.time() > deadline - 5:
            break
        
        # Try different term counts
        found = False
        for terms in range(2, min(8, C + 1)):
            if time.time() > deadline - 5:
                break
            
            result = can_solve(target, 1, C, terms, set())
            if result is not None:
                try_solution(result)
                found = True
                break
        
        if found:
            # Found a solution with max denom <= C
            # Since we're going in order, this is the minimum
            break
    
    # Also try the algebraic approach for 3, 4, 5, 6 terms directly
    # This is faster than the recursive search for small term counts
    
    # 3-term: 1/a + 1/b + 1/c = 5/31
    # c = 31ab / (5ab - 31a - 31b)
    for a in range(1, best_max):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, best_max):
            denom = 5 * a * b - 31 * a - 31 * b
            if denom <= 0:
                continue
            if (31 * a * b) % denom != 0:
                continue
            c = (31 * a * b) // denom
            if c <= b:
                continue
            if c < best_max:
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    best_max = c
                    best_solution = [a, b, c]
    
    # 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
    # For each a, b, solve for c, d
    for a in range(1, best_max):
        if time.time() > deadline - 5:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        p = rem_a.numerator
        q = rem_a.denominator
        b_min = max(a + 1, q // p + 1)
        b_max = min(2 * q // p, best_max - 1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline - 5:
                break
            rem_bc = rem_a - Fraction(1, b)
            if rem_bc <= 0:
                break
            p2 = rem_bc.numerator
            q2 = rem_bc.denominator
            q2_sq = q2 * q2
            c_min = max(b + 1, q2 // p2 + 1)
            
            pairs = _get_factor_pairs(q2_sq)
            for u, v in pairs:
                if time.time() > deadline - 5:
                    break
                if (u + q2) % p2 != 0:
                    continue
                c = (u + q2) // p2
                if c < c_min or c <= b:
                    continue
                if (v + q2) % p2 != 0:
                    continue
                d = (v + q2) // p2
                if d <= c:
                    continue
                if d < best_max:
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                        best_max = d
                        best_solution = [a, b, c, d]
    
    # 5-term search
    for a in range(1, best_max):
        if time.time() > deadline - 5:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        b_min = max(a + 1, rem_a.denominator // rem_a.numerator + 1)
        b_max = min(2 * rem_a.denominator // rem_a.numerator, best_max - 1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline - 5:
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                break
            c_min = max(b + 1, rem_b.denominator // rem_b.numerator + 1)
            c_max = min(3 * rem_b.denominator // rem_b.numerator, best_max - 1)
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline - 5:
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    break
                p = rem_c.numerator
                q = rem_c.denominator
                q2 = q * q
                d_min = max(c + 1, q // p + 1)
                
                pairs = _get_factor_pairs(q2)
                for u, v in pairs:
                    if time.time() > deadline - 5:
                        break
                    if (u + q) % p != 0:
                        continue
                    d = (u + q) // p
                    if d < d_min or d <= c:
                        continue
                    if (v + q) % p != 0:
                        continue
                    e = (v + q) // p
                    if e <= d:
                        continue
                    if e < best_max:
                        if sum(Fraction(1, x) for x in [a, b, c, d, e]) == target:
                            best_max = e
                            best_solution = [a, b, c, d, e]
    
    # Final verification
    s = sum(Fraction(1, d) for d in best_solution)
    assert s == target, f"Sum {s} != {target}"
    assert len(set(best_solution)) == len(best_solution), "Duplicate denominators"
    assert all(1 <= d <= 10000000 for d in best_solution), "Denominator out of range"
    assert 1 <= len(best_solution) <= 256, "Invalid term count"
    
    return best_solution
# EVOLVE-BLOCK-END
