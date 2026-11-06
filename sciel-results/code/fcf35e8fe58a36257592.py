import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    target_frac = Fraction(5, 31)
    target_float = 5.0 / 31.0
    
    best_solution = None
    best_max_den = 10**9

    def dfs(current_sum_frac, current_sum_float, last_den, path, depth):
        nonlocal best_solution, best_max_den
        
        if time.time() > deadline:
            return
            
        remaining_frac = target_frac - current_sum_frac
        remaining_float = target_float - current_sum_float
        
        if abs(remaining_float) < 1e-12:
            if remaining_frac == 0:
                if path:
                    max_d = max(path)
                    if max_d < best_max_den:
                        best_max_den = max_d
                        best_solution = list(path)
            return
            
        if remaining_float < -1e-12:
            return
            
        if depth >= 6:
            return

        # Lower bound for next denominator
        # d >= 1/remaining
        min_d = (remaining_frac.denominator + remaining_frac.numerator - 1) // remaining_frac.numerator
        min_d = max(min_d, last_den + 1)
        
        m = 6 - depth
        
        # Upper bound from best known solution
        upper_d = best_max_den
        
        for d in range(min_d, upper_d + 1):
            if time.time() > deadline:
                return
            
            one_over_d_float = 1.0 / d
            rem_after_float = remaining_float - one_over_d_float
            
            if rem_after_float < -1e-12:
                # Should not happen if min_d is correct, but safe to check
                continue
            
            # Tail max check with floats for pruning
            # Sum of 1/(d+1) to 1/(d+m)
            tail_max_float = 0.0
            for i in range(1, m + 1):
                tail_max_float += 1.0 / (d + i)
            
            if rem_after_float > tail_max_float + 1e-12:
                # If the remaining sum is larger than the max possible sum of the remaining terms,
                # and d is increasing, this condition will persist for larger d.
                break
            
            # Exact fraction check for state update
            rem_after_frac = remaining_frac - Fraction(1, d)
            
            path.append(d)
            dfs(current_sum_frac + Fraction(1, d), current_sum_float + one_over_d_float, d, path, depth + 1)
            path.pop()
            
            # If we found a better solution, best_max_den might have decreased.
            # If current d is now >= best_max_den, no point continuing this loop.
            if d >= best_max_den:
                break

    dfs(Fraction(0), 0.0, 0, [], 0)
    
    if best_solution:
        return best_solution
    else:
        # Fallback to greedy if no solution found in time
        r = target_frac
        vals = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            vals.append(d)
            r -= Fraction(1, d)
        return vals
