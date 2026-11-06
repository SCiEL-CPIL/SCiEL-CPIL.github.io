import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    target = Fraction(5, 31)
    best_max = 1736  # Current known solution max denominator
    
    # We search for a smaller max denominator
    # The solution must use denominators <= M
    # We start from best_max - 1 and go down
    
    # To make the search faster, we can limit the number of terms or use heuristics.
    # The number of terms in the optimal solution is likely small.
    
    def dfs(rem, last_d, path, M):
        nonlocal best_max
        if time.time() > deadline:
            return None
        if rem == 0:
            return path
        if rem < 0:
            return None
            
        # Max possible sum from denominators d > last_d up to M
        # If sum_{d=last_d+1}^M 1/d < rem, we can't reach rem
        # Approximation: ln(M) - ln(last_d)
        if last_d > 0:
            import math
            max_sum_approx = math.log(M) - math.log(last_d)
            if max_sum_approx < rem:
                return None
        
        # Iterate next denominator
        # d must be > last_d
        # 1/d <= rem  => d >= 1/rem
        min_d = max(last_d + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # d must be <= M
        for d in range(min_d, M + 1):
            if time.time() > deadline:
                return None
            next_rem = rem - Fraction(1, d)
            if next_rem < 0:
                # Since d increases, 1/d decreases, so next_rem increases.
                # If next_rem < 0, we need a smaller d? No, we are increasing d.
                # If 1/d > rem, then next_rem < 0.
                # We need 1/d <= rem, so d >= 1/rem.
                # The loop starts at min_d which ensures 1/d <= rem.
                # So this shouldn't happen if min_d is calculated correctly.
                # Wait, min_d = ceil(1/rem). So 1/min_d <= rem.
                # So next_rem >= 0.
                continue
            
            if next_rem == 0:
                return path + [d]
            
            res = dfs(next_rem, d, path + [d], M)
            if res:
                return res
        return None

    # Search for smaller M
    # We can't iterate all the way to 1. 
    # Let's try a range around the current best.
    # Or just check if any M < 1736 works.
    # Given the nature of Egyptian fractions, the "jumps" in max denominator might be significant.
    # Let's try checking M values in descending order.
    
    # Optimization: We can restrict the search to M values that are "likely" to work.
    # But simple descending order is safer.
    
    # To avoid slow searches for M values that clearly don't work,
    # we can stop early if the DFS is too slow.
    
    # Let's try M from 1735 down to 1000.
    # If we find one, we continue to check smaller ones?
    # The task asks to MINIMIZE the largest denominator.
    # So we want the absolute minimum.
    # We should check all M < 1736.
    # But 1736 is small. The search space is small.
    
    # However, DFS for M=1735 might be slow if it explores many branches.
    # We can add a depth limit? No, we don't know the depth.
    # But depth is likely < 10.
    
    # Let's run the search.
    for M in range(1735, 6, -1):
        if time.time() > deadline:
            break
        # Heuristic: if M is very small, it's unlikely to work for 5/31 with few terms.
        # 5/31 ~ 0.16.
        # 1/7 ~ 0.14.
        # If M < 7, no solution.
        
        # Reset DFS
        res = dfs(target, 0, [], M)
        if res:
            best_max = M
            # We found a solution with max M.
            # Can we find a smaller one? Yes, continue loop.
            # But we should verify the solution is valid (distinct, etc).
            # The DFS ensures distinctness and order.
            pass

    # If we didn't find anything smaller, best_max remains 1736.
    # We need to return the actual denominators for the best_max found.
    # If best_max is still 1736, return the known solution.
    # If best_max was updated, we need the solution.
    
    # Let's re-structure to keep the best solution.
    best_solution = [8, 28, 1736]
    
    for M in range(1735, 6, -1):
        if time.time() > deadline:
            break
        res = dfs(target, 0, [], M)
        if res:
            # Verify
            s = sum(Fraction(1, d) for d in res)
            if s == target and len(set(res)) == len(res) and all(1 <= d <= M for d in res):
                best_max = M
                best_solution = res
                # We can't break because we want the MINIMUM M.
                # Continue searching for smaller M.
    
    return best_solution
