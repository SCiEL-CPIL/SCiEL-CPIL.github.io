import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    target = Fraction(5, 31)
    best_max_denom = 10000000  # Initial best is the limit
    
    # Helper to find min max denominator for k terms summing to target
    # d_min: minimum allowed value for the first denominator in this subproblem
    # current_best: current global best max denominator to prune
    def dfs(k, r, d_min, current_best):
        nonlocal best_max_denom
        
        if time.time() > deadline:
            return None
            
        if k == 1:
            # 1/d = r => d = 1/r
            if r.numerator == 1:
                d = r.denominator
                if d >= d_min and d < current_best:
                    return d
            return None
        
        if k == 2:
            # 1/a + 1/b = r
            # a > 1/r, a < 2/r
            a_min = max(d_min, (r.denominator // r.numerator) + 1)
            a_max = (2 * r.denominator) // r.numerator
            # Also a must be < current_best
            a_max = min(a_max, current_best - 1)
            
            for a in range(a_min, a_max + 1):
                rem = r - Fraction(1, a)
                if rem <= 0:
                    continue
                # 1/b = rem => b = 1/rem
                if rem.numerator == 1:
                    b = rem.denominator
                    if b > a and b < current_best:
                        return b
            return None

        # General k
        # 1/d1 + ... + 1/dk = r
        # d1 must be such that 1/d1 < r => d1 > 1/r
        # Also (k)/d1 > r => d1 < k/r
        d1_min = max(d_min, (r.denominator // r.numerator) + 1)
        d1_max = (k * r.denominator) // r.numerator
        d1_max = min(d1_max, current_best - 1)
        
        # Pruning: Sum of smallest k denominators starting from d1_min
        # If this sum < r, then even the smallest possible denominators don't add up enough?
        # Wait. 1/x is decreasing.
        # The *maximum* sum for k terms with min value d1_min is achieved by picking the smallest distinct integers: d1_min, d1_min+1, ...
        # If this max sum < r, then it's impossible.
        max_possible_sum = sum(Fraction(1, d1_min + i) for i in range(k))
        if max_possible_sum < r:
            return None
            
        # The *minimum* sum for k terms with max value < current_best
        # is achieved by picking the largest distinct integers: current_best-1, current_best-2, ...
        # If this min sum > r, then even the largest possible denominators (below best) sum to too much?
        # No, we want sum to be EXACTLY r.
        # If min sum > r, we can't reduce the sum further by increasing denominators (since we are bounded by current_best).
        # Wait, denominators can be larger than current_best? No, we want to improve current_best.
        # So we only consider denominators < current_best.
        # If the sum of the k largest available denominators (< current_best) is > r,
        # it doesn't mean it's impossible, it just means we might need to pick smaller denominators?
        # No, smaller denominators give LARGER sums.
        # So if the sum of the *largest* denominators is > r, we can try to increase denominators to reduce the sum.
        # But we are bounded by current_best.
        # If sum(largest k) > r, it's possible.
        # If sum(largest k) < r, then even the largest denominators don't sum to enough?
        # No, largest denominators give the SMALLEST sum.
        # So if sum(largest k) < r, then it's impossible to reach r with denominators < current_best.
        min_possible_sum = sum(Fraction(1, current_best - 1 - i) for i in range(k))
        if min_possible_sum > r:
            # Actually, if min sum > r, we can't make the sum smaller.
            # So if r is smaller than the minimum possible sum, it's impossible.
            return None
        # Wait, if min sum < r, it's possible.
        # If min sum > r, it's impossible.
        # So: if min_possible_sum > r: return None.
        
        best_in_branch = current_best
        for d1 in range(d1_min, d1_max + 1):
            rem = r - Fraction(1, d1)
            if rem <= 0:
                continue
                
            res = dfs(k - 1, rem, d1 + 1, best_in_branch)
            if res is not None:
                best_in_branch = res
                # If we found a very good solution, update global best
                if res < best_max_denom:
                    best_max_denom = res
                    # If best_max_denom is small, we can prune more
                    if best_max_denom <= d1:
                         break # Can't get smaller max denom with larger d1? 
                         # Actually max denom is res. If res is small, we can stop?
                         # No, we want to find the MIN res.
                         # If we found res, we can prune d1 loops where d1 >= res.
        
        return best_in_branch if best_in_branch < current_best else None

    # Search for k = 3 to 10
    for k in range(3, 11):
        # Estimate initial bound for this k
        # If k terms sum to 5/31, avg denom ~ 31k/5 = 6.2k.
        # Max denom likely larger.
        # Start with a generous bound, e.g., 1000
        initial_bound = 1000
        res = dfs(k, target, 1, initial_bound)
        if res is not None:
            # We found a solution with max denom res.
            # Try to improve? The DFS already tries to minimize.
            # But the initial bound might have pruned good solutions?
            # No, dfs returns the min max denom found within the bound.
            # If res < best_max_denom, update.
            if res < best_max_denom:
                best_max_denom = res
        
        # If we found a very small best_max_denom, we can break?
        # Theoretical lower bound for max denom D:
        # 1/D + 1/(D-1) + ... + 1/(D-k+1) >= 5/31
        # For k=3, D=248.
        # For k=4, D might be smaller.
        # Let's just run all k.
        
        if time.time() > deadline:
            break

    # Fallback to greedy if no better found
    if best_max_denom == 10000000:
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        best_max_denom = max(values)
        return values

    # We need to return the denominators, not just the max.
    # The dfs function above only returned the max denominator value.
    # We need to reconstruct the solution or modify dfs to return the list.
    
    # Let's rewrite dfs to return the list of denominators.
    
    def dfs_full(k, r, d_min, current_best):
        nonlocal best_max_denom
        
        if time.time() > deadline:
            return None
            
        if k == 1:
            if r.numerator == 1:
                d = r.denominator
                if d >= d_min and d < current_best:
                    return [d]
            return None
        
        if k == 2:
            a_min = max(d_min, (r.denominator // r.numerator) + 1)
            a_max = (2 * r.denominator) // r.numerator
            a_max = min(a_max, current_best - 1)
            
            for a in range(a_min, a_max + 1):
                rem = r - Fraction(1, a)
                if rem <= 0:
                    continue
                if rem.numerator == 1:
                    b = rem.denominator
                    if b > a and b < current_best:
                        return [a, b]
            return None

        d1_min = max(d_min, (r.denominator // r.numerator) + 1)
        d1_max = (k * r.denominator) // r.numerator
        d1_max = min(d1_max, current_best - 1)
        
        max_possible_sum = sum(Fraction(1, d1_min + i) for i in range(k))
        if max_possible_sum < r:
            return None
            
        min_possible_sum = sum(Fraction(1, current_best - 1 - i) for i in range(k))
        if min_possible_sum > r:
            return None
        
        best_solution = None
        for d1 in range(d1_min, d1_max + 1):
            rem = r - Fraction(1, d1)
            if rem <= 0:
                continue
                
            res = dfs_full(k - 1, rem, d1 + 1, current_best if best_solution is None else best_solution[-1])
            if res is not None:
                sol = [d1] + res
                if best_solution is None or sol[-1] < best_solution[-1]:
                    best_solution = sol
                    if best_solution[-1] < best_max_denom:
                        best_max_denom = best_solution[-1]
        
        return best_solution

    # Re-run search with full solution tracking
    best_max_denom = 10000000
    best_solution = None
    
    for k in range(3, 11):
        if time.time() > deadline:
            break
        res = dfs_full(k, target, 1, best_max_denom if best_solution else 10000000)
        if res is not None:
            if best_solution is None or res[-1] < best_solution[-1]:
                best_solution = res
                best_max_denom = res[-1]

    if best_solution:
        return best_solution
    else:
        # Fallback
        r = target
        values = []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
