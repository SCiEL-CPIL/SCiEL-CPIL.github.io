import time
from fractions import Fraction

def _find_solution_11(min_den=5, max_den=10000000, deadline=None):
    """
    Find 11 distinct denominators >= min_den whose reciprocal sum is exactly 1.
    Strategy: pick 9 denominators, solve for last 2.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    target = Fraction(1, 1)
    
    def try_solve(dens):
        """Try to solve for last 2 given 9 denominators."""
        current_sum = sum(Fraction(1, d) for d in dens)
        if current_sum >= target:
            return None
        R = target - current_sum
        
        min_a = dens[-1] + 1
        low_a = max(min_a, int(Fraction(1, 1) / R) + 1)
        high_a_frac = Fraction(2, 1) / R
        high_a = min(int(high_a_frac), max_den)
        
        if high_a < low_a:
            return None
        
        for a in range(low_a, high_a + 1):
            Ra_minus_1 = R * a - 1
            if Ra_minus_1 <= 0:
                continue
            b_frac = Fraction(a, 1) / Ra_minus_1
            if b_frac.denominator == 1:
                b = b_frac.numerator
                if b > a and b <= max_den and a not in dens and b not in dens:
                    return dens + [a, b]
        return None
    
    def search(dens, idx):
        if time.time() > deadline:
            return None
        
        if idx == 9:
            return try_solve(dens)
        
        start = dens[idx - 1] + 1 if idx > 0 else min_den
        
        if idx < 3:
            end = start + 30
        elif idx < 6:
            end = start + 50
        else:
            end = start + 100
        
        end = min(end, max_den)
        
        for d in range(start, end + 1):
            if time.time() > deadline:
                return None
            
            partial = sum(Fraction(1, x) for x in dens[:idx]) + Fraction(1, d)
            if partial >= target:
                break
            
            remaining = 9 - idx - 1
            if remaining > 0:
                max_add = sum(Fraction(1, i) for i in range(d + 1, d + 1 + remaining))
                if partial + max_add < target:
                    continue
            
            result = search(dens + [d], idx + 1)
            if result is not None:
                return result
        
        return None
    
    return search([], 0)

def construct():
    deadline = time.time() + 55
    
    res = _find_solution_11(deadline=deadline)
    if res is not None:
        return res
    
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no baseline representation")
