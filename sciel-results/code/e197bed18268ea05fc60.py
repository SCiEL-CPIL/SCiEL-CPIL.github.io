import time
from fractions import Fraction

def construct():
        start_time = time.time()
        deadline = start_time + 55.0  # 5s margin
        
        target = Fraction(1, 1)
        min_den = 5
        max_den = 10000000
        
        # We determined k=8 is the minimum possible.
        # Try k=8 first.
        
        def max_sum(start, count):
            # Max sum of 'count' distinct reciprocals starting from 'start'
            # i.e., 1/start + 1/(start+1) + ...
            s = Fraction(0, 1)
            for i in range(count):
                s += Fraction(1, start + i)
            return s

        def search(start_d, terms_left, rem_sum, current):
            if time.time() > deadline:
                return None
            
            if terms_left == 0:
                if rem_sum == 0:
                    return current
                return None
            
            if terms_left == 1:
                # 1/d = rem_sum => d = 1/rem_sum
                if rem_sum > 0:
                    d = 1 / rem_sum
                    if d.denominator == 1:
                        d_int = d.numerator
                        if d_int > start_d and d_int <= max_den:
                            return current + [d_int]
                return None
            
            # Pruning
            # Max possible sum for terms_left terms starting from start_d + 1
            # Note: The next denominator must be > start_d.
            # The largest sum is using the smallest available denominators.
            max_possible = max_sum(start_d + 1, terms_left)
            if rem_sum > max_possible:
                return None
            
            # Min possible sum?
            # The denominators must be > start_d.
            # To minimize sum, we pick largest denominators.
            # But we are constrained by max_den.
            # This bound is weak.
            
            # Determine range for next denominator d
            # 1/d <= rem_sum => d >= 1/rem_sum
            d_min = max(start_d + 1, int(1 / rem_sum) + 1 if 1 % rem_sum.numerator != 0 else 1 // rem_sum.numerator)
            # Careful with integer division for Fraction.
            # d >= 1/rem_sum.
            # If rem_sum = 1/5, d >= 5.
            # 1/rem_sum = 5.
            
            # Upper bound for d:
            # We need rem_sum - 1/d to be representable by terms_left - 1 terms > d.
            # Max sum of terms_left - 1 terms starting from d+1 is max_sum(d+1, terms_left-1).
            # So we need rem_sum - 1/d <= max_sum(d+1, terms_left-1).
            # Also 1/d < rem_sum is guaranteed by d_min.
            
            # Iterate d
            # We can't iterate to infinity.
            # If d is too large, 1/d is small, so rem_sum - 1/d is close to rem_sum.
            # But max_sum(d+1, terms_left-1) decreases as d increases.
            # So there is a cutoff.
            
            # Let's just iterate and check.
            # To avoid infinite loop, we can break if max_sum(d+1, terms_left-1) < rem_sum - 1/d.
            # Actually, if d is very large, 1/d is small.
            # Let's estimate max d.
            # rem_sum <= max_sum(d+1, terms_left-1) + 1/d
            # For large d, max_sum(d+1, terms_left-1) approx (terms_left-1)/d.
            # rem_sum approx terms_left/d.
            # So d approx terms_left / rem_sum.
            
            d = d_min
            while True:
                if time.time() > deadline:
                    return None
                
                # Check if d is too large
                # The max sum of remaining terms (terms_left - 1) starting from d+1
                if terms_left - 1 > 0:
                    max_rem = max_sum(d + 1, terms_left - 1)
                else:
                    max_rem = Fraction(0, 1)
                
                # We need rem_sum - 1/d <= max_rem
                if rem_sum - Fraction(1, d) > max_rem:
                    break # d is too large, increasing d will only make 1/d smaller and max_rem smaller, so condition won't improve?
                    # Wait.
                    # LHS: rem_sum - 1/d. As d increases, 1/d decreases, so LHS increases.
                    # RHS: max_sum(d+1, ...). As d increases, denominators increase, sum decreases.
                    # So if LHS > RHS now, it will be even worse for larger d.
                    # So we can break.
                
                # Also need rem_sum - 1/d >= 0
                if rem_sum - Fraction(1, d) < 0:
                    # 1/d > rem_sum, so d < 1/rem_sum.
                    # But d >= d_min = ceil(1/rem_sum). So this shouldn't happen if d_min is correct.
                    # But due to floating point or integer logic, maybe.
                    # Actually if d < 1/rem_sum, then 1/d > rem_sum.
                    # We established d_min >= 1/rem_sum.
                    pass
                
                res = search(d, terms_left - 1, rem_sum - Fraction(1, d), current + [d])
                if res:
                    return res
                
                d += 1
                
                # Safety break: if d gets huge, break.
                # Theoretical max d is around terms_left / rem_sum.
                if d > 10 * (terms_left / rem_sum) + 100: 
                     break

        # Try k=8
        res = search(min_den - 1, 8, target, [])
        if res:
            return res
        
        # If k=8 failed (shouldn't), try k=9, etc.
        for k in range(9, 20):
             res = search(min_den - 1, k, target, [])
             if res:
                 return res
             
        raise RuntimeError("No solution found")
