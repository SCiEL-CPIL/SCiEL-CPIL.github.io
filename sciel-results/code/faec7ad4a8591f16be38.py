"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

_START_TIME = time.time()
_TIMEOUT = 85  # safety margin before 90s kill

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

# Prime factorization of 12! = 479001600
# 12! = 2^10 * 3^5 * 5^2 * 7 * 11
_PRIMES = [2, 3, 5, 7, 11]
_EXPONENTS = [10, 5, 2, 1, 1]

def _divisors_from_exp(exp):
    """Generate all divisors from a prime-exponent vector."""
    divs = [1]
    for p, e in zip(_PRIMES, exp):
        new_divs = []
        pw = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    return divs

def _get_divisors_in_range(remaining, min_val, max_val):
    """Generate divisors of remaining in [min_val, max_val] using prime factorization."""
    if remaining == 1:
        return []
    
    # Factor remaining into primes
    exp = [0] * len(_PRIMES)
    rem = remaining
    for i, p in enumerate(_PRIMES):
        while rem % p == 0:
            exp[i] += 1
            rem //= p
    if rem != 1:
        # Should not happen since remaining always divides 12!
        if min_val <= remaining <= max_val:
            return [remaining]
        return []
    
    divs = _divisors_from_exp(exp)
    return [d for d in divs if min_val <= d <= max_val]

def _dfs(remaining, min_val, max_bound, chosen, path, best):
    """
    DFS to find distinct factors in [min_val, max_bound] with product = remaining.
    Returns True if found, updates best.
    """
    if _time_left() < 1:
        return False
    
    if remaining == 1:
        # Found a complete decomposition
        max_factor = max(path) if path else 0
        if max_factor < best['max']:
            best['max'] = max_factor
            best['factors'] = list(path)
        return True
    
    if len(path) >= 12:
        return False
    
    # Get divisors of remaining that are >= min_val and <= max_bound
    divs = _get_divisors_in_range(remaining, min_val, max_bound)
    
    if not divs:
        return False
    
    # Sort to try smaller divisors first (helps find balanced solutions)
    divs.sort()
    
    for d in divs:
        if d in chosen:
            continue
        
        # Branch and bound: if d alone exceeds current best max, skip
        if d >= best['max']:
            continue
        
        new_max = max(d, max(path) if path else 0)
        if new_max >= best['max']:
            continue
        
        rem2 = remaining // d
        
        if rem2 == 1:
            # This is the last factor
            if new_max < best['max']:
                best['max'] = new_max
                best['factors'] = list(path) + [d]
                return True
        else:
            # Need more factors. The next factor must be >= d+1 (for distinctness/ordering)
            # and the remaining rem2 must be factorable
            # Pruning: rem2 must be >= (d+1) if we need at least one more factor
            if rem2 < d + 1:
                continue
            
            # Additional pruning: rem2 must be at least min_val
            if rem2 < min_val:
                continue
            
            chosen.add(d)
            path.append(d)
            
            if _dfs(rem2, d + 1, max_bound, chosen, path, best):
                path.pop()
                chosen.remove(d)
                return True
            
            path.pop()
            chosen.remove(d)
    
    return False

def _search_for_max(M):
    """
    Search for a decomposition of 12! into distinct factors in [13, M].
    Returns the factors if found, None otherwise.
    """
    target = factorial(12)
    best = {'max': M + 1, 'factors': None}
    chosen = set()
    path = []
    
    if _dfs(target, 13, M, chosen, path, best):
        return best['factors']
    return None

def construct():
    target = factorial(12)  # 479001600
    
    # Try to find a good solution by searching M values
    # Start with a reasonable upper bound and work down
    
    # Lower bound: 13
    # Upper bound: target (trivial solution)
    
    # Based on analysis:
    # With k factors all >= 13, product >= 13^k
    # 13^5 = 371293, 13^6 = 4826809, 13^7 = 62748517
    # So we need at least 6 factors for a reasonable M
    
    # Let's try M values from likely ranges
    # M around 100-300 should work with 4-6 factors
    
    # Try a range of M values, starting from smaller ones
    # Use a set of candidate M values that are divisors of 12!
    
    # Get all divisors of 12! that are >= 13 and <= some reasonable bound
    all_divs = _get_divisors_in_range(target, 13, 500)
    all_divs.sort()
    
    best_factors = None
    best_M = target
    
    # Try each candidate M in increasing order
    for M in all_divs:
        if _time_left() < 10:
            break
        
        result = _search_for_max(M)
        if result is not None:
            best_factors = result
            best_M = M
            break  # Found the smallest M that works
    
    if best_factors is not None:
        return best_factors
    
    # Fallback: try a few more M values if we didn't find one
    for M in [100, 150, 200, 250, 300, 400, 500]:
        if _time_left() < 5:
            break
        result = _search_for_max(M)
        if result is not None:
            return result
    
    # Final fallback: return a known valid solution
    # Let's construct one manually
    # 12! = 479001600
    # Try: 15, 20, 24, 30, 35, 44, 55, 88, 165
    # 15*20 = 300, *24 = 7200, *30 = 216000, *35 = 7560000, *44 = 332640000, *55 = 18295200000
    # That's too big. Let me recalculate.
    
    # Better approach: just return the trivial solution if nothing works
    return [target]
# EVOLVE-BLOCK-END
