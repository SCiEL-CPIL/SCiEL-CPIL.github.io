import time
from math import factorial

_START_TIME = time.time()
_TIMEOUT = 85

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def _get_divisors_in_range(n, lo, hi):
    """Generate divisors of n in [lo, hi]."""
    if n == 0:
        return []
    divs = []
    # n is always a divisor of 12!, so n is integer
    limit = int(n**0.5) + 1
    for i in range(1, limit):
        if n % i == 0:
            if lo <= i <= hi:
                divs.append(i)
            j = n // i
            if lo <= j <= hi and j != i:
                divs.append(j)
    return sorted(divs)

def _dfs(rem, min_val, max_val, chosen, path):
    """
    Recursively find factors.
    rem: product remaining
    min_val: smallest factor allowed (for distinctness/ordering)
    max_val: largest factor allowed (the bound M)
    chosen: set of chosen factors
    path: list of factors in current path
    """
    if _time_left() < 1:
        return False
        
    if rem == 1:
        return True
    
    if min_val > max_val:
        return False
        
    if rem < min_val:
        return False
        
    # If rem is prime and > max_val, no solution
    # Simple check: if rem has no divisors in [min_val, max_val], fail
    divs = _get_divisors_in_range(rem, min_val, max_val)
    
    for d in divs:
        if d in chosen:
            continue
        
        chosen.add(d)
        path.append(d)
        
        if _dfs(rem // d, d + 1, max_val, chosen, path):
            return True
        
        path.pop()
        chosen.remove(d)
        
    return False

def construct():
    target = factorial(12)  # 479001600
    
    # We search for the minimal M such that a valid decomposition exists
    # with all factors <= M.
    # Lower bound: 13
    # Upper bound: target (trivial solution [target])
    
    # Based on analysis, M is likely < 200.
    # We iterate M from 13 upwards.
    
    for M in range(13, 1000):
        if _time_left() < 5:
            break
        chosen = set()
        path = []
        if _dfs(target, 13, M, chosen, path):
            return path
            
    # Fallback: return a valid solution if search failed or timed out
    # A trivial solution is [target], but target > 12 is true, distinct is true.
    # However, the problem implies finding a better one.
    # If we reach here, it means no solution with M < 1000 was found,
    # which is highly unlikely.
    return [target]
