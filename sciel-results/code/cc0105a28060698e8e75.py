"""Factorial decompositions into distinct factors above the index.

12! = 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11

Goal: return 1..12 DISTINCT integers all > 12 with product exactly 12!,
minimizing the LARGEST factor.

Strategy:
  1. Enumerate all divisors d of 12! with d > 12.
  2. Use a branch-and-bound / backtracking search over these divisors,
     assigning prime-exponent budgets, to find a subset whose product is 12!
     with the minimum possible maximum element.
  3. Prune aggressively using lower bounds on the remaining product.
"""
import math
import time

T12 = math.factorial(12)  # 479001600
PRIMES = [2, 3, 5, 7, 11]
EXP_TOTAL = {2: 10, 3: 5, 5: 2, 7: 1, 11: 1}

def _exp_vector(n):
    """Return tuple of prime exponents for n (only primes in PRIMES matter)."""
    v = [0] * len(PRIMES)
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            n //= p
            v[i] += 1
    return tuple(v)

def _build_divisors():
    """Return sorted list of divisors of T12 that are > 12, with their exponent vectors."""
    divs = []
    limit = math.isqrt(T12)
    seen = set()
    for i in range(1, limit + 1):
        if T12 % i == 0:
            for d in (i, T12 // i):
                if d > 12 and d not in seen:
                    seen.add(d)
                    divs.append(d)
    divs.sort()
    return divs

def _valid_solution(factors):
    if not (1 <= len(factors) <= 12):
        return False
    if len(set(factors)) != len(factors):
        return False
    if any(f < 13 for f in factors):
        return False
    p = 1
    for f in factors:
        p *= f
    return p == T12

def _search(divs, exp_vecs, deadline):
    """
    Backtracking search to find a subset of divs whose exponent sum equals EXP_TOTAL,
    minimizing the maximum divisor in the subset.
    
    We search for the minimum possible max factor M.
    Approach: binary search on M, or direct branch-and-bound.
    Here we do branch-and-bound: try to build a valid set while tracking the max.
    
    Actually, a cleaner approach: iterate candidate max values in increasing order,
    and for each, check if there exists a valid subset using only divisors <= that max.
    The first such max is optimal.
    """
    # Sort divisors; we'll try max_bound in increasing order
    # But that could be slow. Instead, do a DFS that minimizes max.
    
    # Precompute: for each divisor, its exponent vector (already done)
    # DFS state: index into divs, remaining exponents, current max, chosen factors
    # To minimize max, we want to find any valid solution, then try to improve.
    
    # Better: branch and bound where we sort divisors and use the fact that
    # all chosen divisors must be <= current best max.
    
    best_max = T12  # worst case: single factor
    best_factors = [T12]
    
    n = len(divs)
    # We'll do a DFS that picks divisors in sorted order, ensuring distinctness by construction
    # (each divisor can be used at most once).
    # State: pos (index in divs), rem_exp (tuple of remaining exponents), chosen (list)
    # Pruning: if the product of remaining exponents can't be achieved, stop.
    
    # Lower bound on max: at least 13. Upper bound: current best_max.
    # We can also compute: the product of the largest possible remaining factors
    # must be at least the remaining product.
    
    def dfs(pos, rem_exp, chosen):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        # Check if all exponents are zero
        if all(e == 0 for e in rem_exp):
            if chosen:
                m = max(chosen)
                if m < best_max:
                    best_max = m
                    best_factors = list(chosen)
            return
        
        # If we've run out of divisors
        if pos >= n:
            return
        
        # Pruning: can we still achieve the remaining exponents with divisors from pos onward?
        # Quick check: the remaining product must be achievable.
        # Compute remaining product from exponents
        rem_prod = 1
        for i, p in enumerate(PRIMES):
            if rem_exp[i] > 0:
                rem_prod *= p ** rem_exp[i]
        
        # The product of all remaining divisors from pos to end must be >= rem_prod
        # But that's expensive. Instead, a simpler bound:
        # The minimum number of factors needed: at least 1 (if rem_prod is a divisor)
        # The maximum factor in any solution from here must be >= rem_prod / (product of all smaller available)
        # This is complex. Let's use a simpler heuristic:
        # If rem_prod > 1, we need at least one more factor >= some value.
        
        # Try skipping divisor at pos
        dfs(pos + 1, rem_exp, chosen)
        
        # Try taking divisor at pos
        d = divs[pos]
        if d >= best_max:
            # Taking this would not improve best_max (since d >= best_max)
            # But it might be equal; still no improvement. Skip.
            # Actually if d == best_max, it doesn't improve, so skip.
            pass
        else:
            ev = exp_vecs[pos]
            # Check if taking this divisor is feasible (doesn't exceed remaining exponents)
            feasible = True
            for i in range(len(PRIMES)):
                if ev[i] > rem_exp[i]:
                    feasible = False
                    break
            if feasible:
                new_rem = tuple(rem_exp[i] - ev[i] for i in range(len(PRIMES)))
                chosen.append(d)
                dfs(pos + 1, new_rem, chosen)
                chosen.pop()
    
    start_exp = tuple(EXP_TOTAL[p] for p in PRIMES)
    dfs(0, start_exp, [])
    
    return best_factors, best_max

def _local_improve(factors, deadline):
    """Try to reduce the max factor by splitting it into two smaller distinct factors."""
    if not factors:
        return factors
    factors = sorted(factors)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur_max = factors[-1]
        others = factors[:-1]
        for a in range(13, math.isqrt(cur_max) + 1):
            if time.time() > deadline:
                break
            if cur_max % a == 0:
                b = cur_max // a
                if b > a and a >= 13 and b >= 13:
                    if a not in others and b not in others:
                        new_factors = sorted(others + [a, b])
                        if _valid_solution(new_factors):
                            if max(new_factors) < cur_max:
                                factors = new_factors
                                improved = True
                                break
    return factors

def construct():
    deadline = time.time() + 80  # safety margin
    
    divs = _build_divisors()
    exp_vecs = [_exp_vector(d) for d in divs]
    
    # Run the main search
    best_factors, best_max = _search(divs, exp_vecs, deadline)
    
    # Local improvement
    if best_factors and time.time() < deadline:
        best_factors = _local_improve(best_factors, deadline)
    
    # Verify
    if best_factors and _valid_solution(best_factors):
        return best_factors
    
    # Emergency fallback
    return [T12]
