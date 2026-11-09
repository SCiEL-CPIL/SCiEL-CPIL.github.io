from fractions import Fraction
import math

def construct():
    # Known minimum is 8.
    # We search for k=8 first.
    
    def find_solution(k):
        # Use a stack to avoid recursion limits if needed, but k is small.
        # State: (current_sum, next_min_d, rem, path)
        
        def dfs(current_sum, next_d, rem, path):
            if rem == 0:
                if current_sum == 1:
                    return path
                return None
            
            deficit = 1 - current_sum
            if deficit <= 0:
                return None
            
            # Lower bound for d
            # 1/d <= deficit => d >= 1/deficit
            min_d_from_deficit = (deficit.denominator + deficit.numerator - 1) // deficit.numerator
            min_d = max(next_d, min_d_from_deficit)
            
            # Upper bound for d
            # Sum of rem terms starting from d must be >= deficit
            # rem/d >= deficit => d <= rem/deficit
            max_d = int(rem / deficit) + 1
            
            if min_d > max_d:
                return None
                
            # Iterate d
            # Heuristic: try smaller d first? Or larger?
            # Smaller d gives larger 1/d, closer to 1 quickly.
            for d in range(min_d, max_d + 1):
                # Pruning: max possible sum with d, d+1...
                # Approx rem/d
                if rem / d < deficit:
                    break
                
                new_sum = current_sum + Fraction(1, d)
                if new_sum > 1:
                    continue # Since d increases, 1/d decreases, but we started from min_d where 1/d <= deficit.
                    # Actually if min_d was calculated correctly, new_sum <= 1.
                    
                res = dfs(new_sum, d + 1, rem - 1, path + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(Fraction(0, 1), 5, k, [])

    for k in range(8, 12): # Only search up to 11 to save time
        sol = find_solution(k)
        if sol:
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                return sol
    raise RuntimeError("No solution found")
