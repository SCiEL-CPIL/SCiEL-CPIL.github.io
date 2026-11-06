from fractions import Fraction
import time
import math

def construct():
    deadline = time.time() + 50
    
    # Greedy solution as baseline
    def greedy_solve():
        n, d = 1, 1
        result = []
        min_d = 5
        while n > 0:
            # Find smallest x >= min_d such that 1/x <= n/d
            # x >= d/n
            x = (d + n - 1) // n
            if x < min_d:
                x = min_d
            
            # Ensure 1/x <= n/d
            # If x < d/n, 1/x > n/d. But we took ceil, so x >= d/n.
            # Check distinctness
            if x in result:
                # Find next valid x
                while x in result or 1/x > n/d:
                    x += 1
            
            result.append(x)
            # new rem = n/d - 1/x = (nx - d) / dx
            new_n = n * x - d
            new_d = d * x
            # Reduce fraction
            g = math.gcd(new_n, new_d)
            n = new_n // g
            d = new_d // g
            min_d = x + 1
            
            if len(result) > 256:
                return None # Should not happen for 1, but safety
        return result

    best = greedy_solve()
    
    # DFS to find shorter solution
    # State: (n, d, min_d, current_list)
    # Target: n=0
    # Pruning:
    # 1. If len(current) >= len(best), return
    # 2. If n/d > max_possible_sum(remaining_terms, min_d), return
    #    Max sum of k distinct unit fractions with denominators > min_d
    #    is less than k / (min_d + 1).
    # 3. If n/d < min_possible_sum? No, min sum is 0.
    #    But we need to be able to form n/d.
    #    Lower bound on next denominator x: ceil(d/n).
    #    Upper bound on next denominator x:
    #    We need n/d - 1/x <= (k-1) / (x+1)
    #    n/d <= 1/x + (k-1)/(x+1)
    #    For large x, n/d <= k/x => x <= k*d/n.
    #    So we can limit x to around k*d/n.

    def dfs(n, d, min_d, current, depth):
        nonlocal best, deadline
        
        if time.time() > deadline:
            return
        
        if n == 0:
            if len(current) < len(best):
                best = list(current)
            return
        
        if len(current) >= len(best):
            return
            
        remaining_terms = len(best) - len(current)
        
        # Pruning: Max possible sum with remaining_terms terms starting from min_d
        # Sum < remaining_terms / (min_d + 1)
        if n * (min_d + 1) > d * remaining_terms:
            return
            
        # Next denominator x
        # x >= ceil(d/n)
        x_min = (d + n - 1) // n
        if x_min < min_d:
            x_min = min_d
            
        # x must be distinct from current
        # Upper bound for x
        # n/d <= 1/x + (remaining_terms - 1) / (x + 1)
        # Approx x <= remaining_terms * d / n
        x_max = int((remaining_terms * d) // n) + 2
        if x_max > 10000000:
            x_max = 10000000
            
        # Also, if x is too large, 1/x is too small to reduce n/d significantly?
        # Actually, if x is very large, the remainder is almost n/d.
        # We just iterate x from x_min to x_max.
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                return
            
            if x in current:
                continue
                
            # Check 1/x <= n/d
            if x < d / n: # Using float for quick check, or x*n < d
                if x * n < d:
                    continue
                else:
                    # x*n >= d, so 1/x <= n/d
                    pass
            
            # new rem = n/d - 1/x = (nx - d) / dx
            new_n = n * x - d
            if new_n < 0:
                continue
            if new_n == 0:
                if len(current) + 1 < len(best):
                    best = list(current) + [x]
                return # Found a solution, no need to explore deeper for this branch? 
                       # Actually, we want to minimize count. If we found one, we can update best.
                       # But there might be a shorter one? No, we are at depth+1.
                       # If we found a solution of length L, we can prune branches >= L.
                       # Since we update best, the pruning condition `len(current) >= len(best)` handles it.
                       # But we should continue searching for shorter? 
                       # If we found a solution of length L, any other solution of length L is not better.
                       # We want shorter. So we can return from this branch.
            
            # Reduce new_n / new_d
            new_d = d * x
            g = math.gcd(new_n, new_d)
            rn = new_n // g
            rd = new_d // g
            
            current.append(x)
            dfs(rn, rd, x + 1, current, depth + 1)
            current.pop()
            
    # Run DFS with a short time budget to find a better solution
    # We try to find a solution with <= len(best) - 1 terms
    # Start with depth 0
    # To avoid explosion, we limit the search to finding a solution of length <= 7 or 8
    # If greedy gave 10, we try to find 9, 8, etc.
    # The DFS above uses `best` to prune.
    # We call it.
    
    # Optimization: Only run DFS if we have time.
    if time.time() < deadline - 10:
        dfs(1, 1, 5, [], 0)
        
    return best
