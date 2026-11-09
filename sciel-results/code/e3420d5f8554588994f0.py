from fractions import Fraction

def construct():
    # We know the minimum number of terms is 8.
    # We will search for a solution with 8 terms first.
    # If not found, we try 9, 10, etc.
    
    def find_solution(k):
        # DFS with pruning
        # state: (current_sum, next_min_denom, remaining_terms)
        # We use a recursive function or a stack.
        
        # To avoid recursion depth issues and overhead, we can use a stack or recursion.
        # k is small (<= 10 likely), so recursion is fine.
        
        def dfs(current_sum, next_d, rem, path):
            if rem == 0:
                if current_sum == 1:
                    return path
                return None
            
            deficit = 1 - current_sum
            if deficit <= 0:
                return None
            
            # Lower bound for d: 1/d <= deficit
            # d >= ceil(1/deficit)
            # 1/deficit = deficit.denominator / deficit.numerator
            min_d_from_deficit = (deficit.denominator + deficit.numerator - 1) // deficit.numerator
            min_d = max(next_d, min_d_from_deficit)
            
            # Upper bound for d:
            # The sum of the remaining 'rem' terms, using the smallest possible denominators
            # starting from d, is sum(1/(d+i) for i in range(rem)).
            # This sum must be >= deficit.
            # A loose upper bound: rem/d >= deficit => d <= rem/deficit
            # So we can stop if d > rem/deficit
            max_d = int(rem / deficit) + 1
            
            if min_d > max_d:
                return None
                
            for d in range(min_d, max_d + 1):
                # Pruning: check if max possible sum with d, d+1... is >= deficit
                # Approx: rem/d
                # If rem/d < deficit, then even if all were 1/d, sum is too small.
                # Since denominators are distinct and increasing, actual sum < rem/d.
                # So if rem/d < deficit, break.
                if rem / d < deficit:
                    break
                
                new_sum = current_sum + Fraction(1, d)
                # If new_sum > 1, break (since d increases, 1/d decreases, but we are in a loop)
                # Actually, if new_sum > 1, then for larger d, 1/d is smaller, so new_sum might become <= 1?
                # No, if 1/d > deficit, then new_sum > 1.
                # We already enforced d >= ceil(1/deficit), so 1/d <= deficit.
                # So new_sum <= 1.
                
                res = dfs(new_sum, d + 1, rem - 1, path + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(Fraction(0, 1), 5, k, [])

    for k in range(8, 257):
        sol = find_solution(k)
        if sol:
            # Verify solution
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                return sol
    raise RuntimeError("No solution found")
