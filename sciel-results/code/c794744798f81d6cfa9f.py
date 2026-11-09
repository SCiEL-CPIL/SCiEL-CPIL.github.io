import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 85.0  # Safety margin from 90s limit

def _verify(denoms):
    """Verify a set of denominators sums to TARGET and meets constraints."""
    if not denoms:
        return False
    if len(denoms) > MAX_TERMS:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return False
    s = sum(Fraction(1, d) for d in denoms)
    return s == TARGET

def _find_3_term_solution(max_denom):
    """
    Find a 3-term Egyptian fraction 1/a + 1/b + 1/c = 5/31
    with max(a,b,c) < max_denom.
    Returns (a, b, c) or None.
    """
    # We need 1/a + 1/b + 1/c = 5/31
    # WLOG a <= b <= c
    # a must be >= ceil(31/5) = 7
    # For a given a, we need 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a)
    # Let num = 5a - 31, den = 31a
    # We need 1/b + 1/c = num/den where num > 0
    # (den - b)(den - c) = den^2 / num ... actually:
    # 1/b + 1/c = num/den => (b+c)/(bc) = num/den => den(b+c) = num*bc
    # num*bc - den*b - den*c = 0
    # (num*b - den)(num*c - den) = den^2
    # So for each divisor d of den^2, we can set:
    # num*b - den = d => b = (d + den) / num
    # num*c - den = den^2/d => c = (den^2/d + den) / num
    
    for a in range(7, max_denom):
        num = 5 * a - 31
        if num <= 0:
            continue
        den = 31 * a
        
        # b must be >= a, and 1/b + 1/c = num/den
        # b >= max(a, ceil(den/num))
        b_min = max(a, (den + num - 1) // num)
        # c >= b, and 1/b + 1/c <= 2/b => num/den <= 2/b => b <= 2*den/num
        b_max = (2 * den + num - 1) // num
        
        if b_min > b_max:
            continue
        
        # For each valid b, check if c is an integer
        # 1/c = num/den - 1/b = (num*b - den) / (den*b)
        # c = den*b / (num*b - den)
        # Need num*b - den > 0 and (num*b - den) | den*b
        
        for b in range(b_min, b_max + 1):
            # Check if c is integer
            # c = den*b / (num*b - den)
            denom_c = num * b - den
            if denom_c <= 0:
                continue
            if (den * b) % denom_c != 0:
                continue
            c = (den * b) // denom_c
            if c < b:
                continue
            if c >= max_denom:
                break
            # Check distinctness
            if a == b or b == c or a == c:
                continue
            # Verify
            if _verify([a, b, c]):
                return [a, b, c]
    
    return None

def _dfs(remaining, min_d, used, current_max, cap, best_holder, start_time):
    """
    DFS to find a solution with max denominator < cap.
    """
    if time.time() > start_time + DEADLINE - 2:
        return False
    
    if remaining == 0:
        if current_max < best_holder['val']:
            best_holder['val'] = current_max
            best_holder['sol'] = list(used)
        return True
    
    if current_max >= cap:
        return False
        
    # Pruning: If remaining is too small to be formed by any number of terms >= min_d
    if remaining < Fraction(1, cap):
        return False

    # Pruning: If remaining is too large
    k = MAX_TERMS - len(used)
    if k <= 0:
        return False
        
    max_sum = Fraction(0, 1)
    for i in range(k):
        max_sum += Fraction(1, min_d + i)
        
    if remaining > max_sum:
        return False

    # Lower bound for d: 1/d <= remaining => d >= ceil(1/remaining)
    inv_rem = 1 / remaining
    min_d_eff = max(min_d, (inv_rem.numerator + inv_rem.denominator - 1) // inv_rem.denominator)
    
    for d in range(min_d_eff, cap + 1):
        if time.time() > start_time + DEADLINE - 2:
            return False
            
        if d in used:
            continue
            
        term = Fraction(1, d)
        if term > remaining:
            continue
            
        new_rem = remaining - term
        if new_rem < 0:
            continue
            
        if d > current_max:
            new_max = d
        else:
            new_max = current_max
            
        # Pruning: Can we form new_rem with denominators in [d+1, cap]?
        if new_rem > 0 and new_rem < Fraction(1, cap):
            continue
                
        used.add(d)
        if _dfs(new_rem, d + 1, used, new_max, cap, best_holder, start_time):
            used.remove(d)
            return True
        used.remove(d)
        
    return False

def construct():
    start_time = time.time()
    
    # Known solution
    best_sol = [8, 31, 248]
    best_max = 248
    
    # Try to find better 3-term solution
    sol = _find_3_term_solution(best_max)
    
    if sol:
        best_sol = sol
        best_max = max(sol)
    
    # Try to find 4+ term solution with smaller max denominator
    # Use DFS with cap = best_max - 1
    if time.time() - start_time < DEADLINE - 10 and best_max > 100:
        best_holder = {'val': best_max, 'sol': best_sol}
        _dfs(TARGET, 7, set(), 0, best_max - 1, best_holder, start_time)
        
        if best_holder['sol']:
            best_sol = best_holder['sol']
            best_max = best_holder['val']
    
    return best_sol
