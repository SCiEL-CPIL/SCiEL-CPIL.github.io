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
        
        # R is a Fraction.
        # We need 1/a + 1/b = R with a < b.
        # b = a / (R*a - 1)
        # For b to be integer, R*a - 1 must divide a.
        # Also b > a implies a / (R*a - 1) > a => 1 / (R*a - 1) > 1 => R*a - 1 < 1 => R*a < 2 => a < 2/R.
        # And R*a - 1 > 0 => a > 1/R.
        
        # Tighter bound for a_max using b >= a+1
        # R <= 1/a + 1/(a+1)
        # R a^2 + (R-2)a - 1 <= 0
        # a_max = (-(R-2) + sqrt((R-2)^2 + 4R)) / (2R)
        
        # Compute bounds using integers to avoid float precision issues if possible, 
        # but Fraction is exact.
        # Let's calculate a_max_root
        # (2-R)^2 + 4R = 4 - 4R + R^2 + 4R = R^2 + 4
        # a_max = (2-R + sqrt(R^2+4)) / (2R)
        
        # We can approximate or just use the 2/R bound if calculation is heavy, 
        # but R is Fraction, so we can do exact integer math for the check.
        
        low_a_frac = Fraction(1, 1) / R
        high_a_frac = Fraction(2, 1) / R
        
        low_a = max(min_a, low_a_frac.numerator // low_a_frac.denominator + 1)
        high_a = min(high_a_frac.numerator // high_a_frac.denominator, max_den)
        
        if high_a < low_a:
            return None
        
        dens_set = set(dens)
        for a in range(low_a, high_a + 1):
            if a in dens_set:
                continue
            Ra_minus_1 = R * a - 1
            if Ra_minus_1 <= 0:
                continue
            b_frac = Fraction(a, 1) / Ra_minus_1
            if b_frac.denominator == 1:
                b = b_frac.numerator
                if b > a and b <= max_den and b not in dens_set:
                    return dens + [a, b]
        return None
    
    def search(dens, idx, partial_sum):
        if time.time() > deadline:
            return None
        
        if idx == 9:
            return try_solve(dens)
        
        start = dens[idx - 1] + 1 if idx > 0 else min_den
        
        remaining = 9 - idx
        if remaining > 0:
            # Max sum of remaining terms
            max_add = sum(Fraction(1, i) for i in range(start, start + remaining))
            if partial_sum + max_add < target:
                return None
        
        # Heuristic bounds for the next denominator
        # Estimate ideal d: 1/d ~ R_rem / remaining
        # R_rem = target - partial_sum
        R_rem = target - partial_sum
        # d ~ remaining / R_rem
        ideal_d = (remaining * R_rem.denominator) // R_rem.numerator
        if ideal_d < start:
            ideal_d = start
            
        # Search window around ideal_d
        # We want to allow some flexibility
        # If we pick d too small, R becomes small, interval for 'a' in try_solve becomes large.
        # If we pick d too large, we might not reach target.
        # So we should probably search d >= ideal_d?
        # Or just a window.
        
        # Let's try a window [start, start + 100] but biased towards larger d?
        # Actually, smaller d (larger 1/d) is "safer" for reaching the sum, but "harder" for the final step.
        # Larger d (smaller 1/d) is "risky" for reaching sum, but "easier" for final step.
        # Given we have 9 slots, we have plenty of room.
        
        # Let's stick to a moderate window but ensure we don't miss small d.
        # Previous code used 30, 50, 100.
        # Let's use a wider window for early indices to allow larger denominators.
        
        if idx < 3:
            end = start + 50
        elif idx < 6:
            end = start + 80
        else:
            end = start + 150
            
        end = min(end, max_den)
        
        for d in range(start, end + 1):
            if time.time() > deadline:
                return None
            
            new_partial = partial_sum + Fraction(1, d)
            if new_partial >= target:
                break
            
            # Additional pruning: check if new_partial + max_possible_remaining_2 < target
            rem_2 = 9 - (idx + 1)
            if rem_2 > 0:
                max_add_2 = sum(Fraction(1, i) for i in range(d + 1, d + 1 + rem_2))
                if new_partial + max_add_2 < target:
                    continue
            
            result = search(dens + [d], idx + 1, new_partial)
            if result is not None:
                return result
        
        return None
    
    return search([], 0, Fraction(0, 1))

def construct():
    deadline = time.time() + 55
    
    res = _find_solution_11(deadline=deadline)
    if res is not None:
        return res
    
    # Fallback to seed logic (2520 divisors)
    # This is guaranteed to find a solution, but might have many terms.
    # The task asks to minimize count.
    # If we fail to find 11, we return the fallback.
    # The fallback in the seed program finds a subset of divisors of 2520 summing to 2520 (which means sum of 1/d = 1).
    # This is a valid solution, but likely has more than 11 terms.
    
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
