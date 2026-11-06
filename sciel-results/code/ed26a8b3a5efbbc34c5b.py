"""Rewritten program: search geometric progressions for minimal max(|A+A|,|A*A|)."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import product
from fractions import Fraction

# Time budget with safety margin
DEADLINE = 85.0  # seconds

def _eval_set(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    n = len(A)
    # Sumset A+A with equal summands allowed
    sums = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
    # Product set A*A with equal factors allowed
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _eval_set_fast(A):
    """Vectorized evaluation using numpy for speed."""
    import numpy as np
    arr = np.array(A, dtype=np.int64)
    n = len(arr)
    # Upper triangle indices
    iu, ju = np.triu_indices(n)
    sums = arr[iu] + arr[ju]
    prods = arr[iu] * arr[ju]
    return max(len(np.unique(sums)), len(np.unique(prods)))

def _try_geometric(a, p, q, n=20, max_val=10000):
    """Try geometric progression a*(p/q)^i. Returns (set, score) or (None, inf)."""
    if a <= 0 or p <= 0 or q <= 0:
        return None, float('inf')
    # Generate terms
    terms = []
    val = Fraction(a, 1)
    for i in range(n):
        # Check if val is integer
        if val.denominator != 1:
            # Try to round to nearest integer? No, must be exact integer.
            # Actually, for a geometric progression with rational ratio p/q,
            # terms are a*p^i/q^i. For these to be integers, q^i must divide a*p^i.
            # This is restrictive. Let's use a different approach:
            # Generate as integers directly if possible, or skip.
            pass
        terms.append(val.numerator // val.denominator if val.denominator == 1 else None)
        if terms[-1] is None:
            return None, float('inf')
        val *= Fraction(p, q)
    
    # Check all are positive integers <= max_val
    for t in terms:
        if t is None or t <= 0 or t > max_val:
            return None, float('inf')
    
    # Check distinct
    if len(set(terms)) != n:
        return None, float('inf')
    
    terms.sort()
    return terms, _eval_set_fast(terms)

def _try_geometric_int(a, ratio_num, ratio_den, n=20, max_val=10000):
    """
    Generate geometric progression where each term is integer.
    term_i = a * ratio_num^i // ratio_den^i (if divisible)
    """
    terms = []
    for i in range(n):
        num = a * (ratio_num ** i)
        den = ratio_den ** i
        if num % den != 0:
            return None
        t = num // den
        if t <= 0 or t > max_val:
            return None
        terms.append(t)
    
    if len(set(terms)) != n:
        return None
    terms.sort()
    return terms

def construct():
    start_time = time.time()
    best_set = list(range(1, 21))
    best_score = _eval_set_fast(best_set)
    
    # Strategy: Search over geometric progressions with rational ratios p/q
    # where p and q are small integers.
    # term_i = a * p^i / q^i, must be integer.
    
    # Precompute powers for speed
    max_pq = 12
    powers = {}
    for base in range(1, max_pq + 1):
        pw = [1]
        for _ in range(20):
            pw.append(pw[-1] * base)
        powers[base] = pw
    
    # Search over a (base value), p, q
    # To keep terms <= 10000, a * (p/q)^19 <= 10000
    # If p > q, terms grow. If p < q, terms shrink.
    
    # Case 1: p > q (increasing sequence)
    # a * (p/q)^19 <= 10000 => a <= 10000 * (q/p)^19
    # Since p > q, (q/p) < 1, so a is bounded.
    
    # Case 2: p < q (decreasing sequence)
    # a <= 10000, and a*(p/q)^19 >= 1 => a >= (q/p)^19
    # But a must be integer.
    
    # Let's iterate over small p, q coprime pairs.
    tested = set()
    
    for p in range(1, max_pq + 1):
        for q in range(1, max_pq + 1):
            if math.gcd(p, q) != 1:
                continue
            if (p, q) in tested:
                continue
            tested.add((p, q))
            
            # Determine range of a
            # We need a * p^i / q^i to be integer for all i in [0, 19]
            # This implies q^19 must divide a * p^19. Since gcd(p,q)=1, q^19 must divide a.
            # So a = k * q^19 for some integer k >= 1.
            # Also a * p^19 / q^19 = k * p^19 <= 10000 (for increasing case p>q)
            # Or a = k * q^19 <= 10000 (for decreasing case p<q, since p^i/q^i < 1)
            
            q19 = q ** 19
            p19 = p ** 19
            
            if p > q:
                # Increasing: a = k * q^19, term_19 = k * p^19 <= 10000
                # k <= 10000 // p^19
                max_k = 10000 // p19
                if max_k < 1:
                    continue
                for k in range(1, max_k + 1):
                    a = k * q19
                    if a > 10000:
                        continue
                    terms = _try_geometric_int(a, p, q)
                    if terms is not None:
                        if time.time() > DEADLINE:
                            return best_set
                        score = _eval_set_fast(terms)
                        if score < best_score:
                            best_score = score
                            best_set = terms
            elif p < q:
                # Decreasing: a = k * q^19, a <= 10000
                # k * q^19 <= 10000 => k <= 10000 // q^19
                max_k = 10000 // q19
                if max_k < 1:
                    continue
                for k in range(1, max_k + 1):
                    a = k * q19
                    if a > 10000:
                        continue
                    terms = _try_geometric_int(a, p, q)
                    if terms is not None:
                        if time.time() > DEADLINE:
                            return best_set
                        score = _eval_set_fast(terms)
                        if score < best_score:
                            best_score = score
                            best_set = terms
            else:
                # p == q == 1, constant sequence, not distinct
                continue
            
            if time.time() > DEADLINE:
                return best_set
    
    # If we have time, try some random local searches or other structures
    # But for now, return best found
    return best_set

# EVOLVE-BLOCK-END
