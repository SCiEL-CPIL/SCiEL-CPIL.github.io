"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find distinct denominators in [1, 10^7] summing reciprocally to exactly 5/31.
    Minimize the largest denominator.
    """
    target = Fraction(5, 31)
    max_time = 80.0  # Safety margin before 90s kill
    start_time = time.time()
    
    best_max_denom = float('inf')
    best_solution = None
    current_solution = None  # Fallback to seed if no better found

    # 1. Try to find a solution with small max denominator using DFS with pruning.
    # We search for solutions with max denominator <= D.
    # We can increase D until we find a solution or run out of time.
    
    # Heuristic: The seed solution has max denom 25? Let's check seed.
    # Seed: 5/31 = 1/8 + ...
    # 5/31 - 1/8 = (40-31)/248 = 9/248
    # 9/248 -> 1/28 (248/9 = 27.55 -> 28)
    # 9/248 - 1/28 = (9*28 - 248) / (248*28) = (252-248)/6944 = 4/6944 = 1/1736
    # So seed solution is [8, 28, 1736]. Max denom is 1736.
    # Can we do better?
    
    # Let's try to find a solution with max denom < 1736.
    # We can use a recursive search.
    
    def dfs(rem, min_denom, current_denoms, max_denom_limit):
        nonlocal best_max_denom, best_solution, current_solution
        if time.time() - start_time > max_time:
            return False
        
        if rem == 0:
            if len(current_denoms) > 0:
                max_d = max(current_denoms)
                if max_d < best_max_denom:
                    best_max_denom = max_d
                    best_solution = list(current_denoms)
                    return True
            return False
        
        if rem < 0:
            return False
            
        # If rem is positive, we need at least one more term.
        # The next term 1/d must be <= rem, so d >= 1/rem.
        # Also d >= min_denom to ensure distinctness and order.
        # Also d <= max_denom_limit.
        
        lower_bound = max(min_denom, (rem.denominator + rem.numerator - 1) // rem.numerator)
        if lower_bound > max_denom_limit:
            return False
            
        # Upper bound on d:
        # If we use 1/d, the remaining sum is rem - 1/d.
        # The remaining sum must be representable by terms with denominators > d.
        # A loose upper bound: if we have k terms left, the max sum is roughly k/d.
        # But we don't know k.
        # A tighter bound: 1/d <= rem => d >= 1/rem.
        # Also, if we pick d too large, rem - 1/d might be too small to be formed by subsequent terms with denominators > d.
        # Specifically, if rem - 1/d > 0, the next term 1/d2 must satisfy d2 > d.
        # So rem - 1/d <= 1/(d+1) is a necessary condition? No, that's for the next single term.
        # Actually, if rem - 1/d > 0, we need to form it with denominators > d.
        # The maximum possible sum of distinct unit fractions with denominators > d is roughly log(d) or similar? No, it converges.
        # But for a small number of terms, the sum is bounded by 1/(d+1) + 1/(d+2) + ...
        # Let's just iterate d from lower_bound up to max_denom_limit.
        
        for d in range(lower_bound, max_denom_limit + 1):
            if time.time() - start_time > max_time:
                return False
                
            next_rem = rem - Fraction(1, d)
            if next_rem < 0:
                break # Since d is increasing, 1/d is decreasing, so next_rem is increasing?
                # Wait. rem is fixed. d increases -> 1/d decreases -> rem - 1/d increases.
                # So if next_rem < 0, it means 1/d > rem.
                # As d increases, 1/d decreases, so eventually 1/d <= rem.
                # So we should CONTINUE, not break.
                # Let's re-evaluate.
                # We start d at lower_bound which ensures 1/d <= rem.
                # So next_rem >= 0 always in the loop range.
                # So this check is redundant if lower_bound is correct.
                pass
            
            current_denoms.append(d)
            if dfs(next_rem, d + 1, current_denoms, max_denom_limit):
                current_denoms.pop()
                return True
            current_denoms.pop()
            
        return False

    # Start with the seed solution's max denom as the initial limit.
    # Seed: [8, 28, 1736]. Max is 1736.
    # Let's try to find something with max denom < 1736.
    # We can try limits like 100, 200, 500, 1000, 1500.
    
    limits_to_try = [10, 20, 50, 100, 200, 500, 1000, 1500, 1735]
    
    for limit in limits_to_try:
        if time.time() - start_time > max_time:
            break
        # Reset for this limit search?
        # We want to find ANY solution with max denom <= limit.
        # If we find one, we update best.
        # Then we can try a smaller limit?
        # Actually, we should try smaller limits first to find the minimum.
        # But the search space is larger for smaller limits? No, smaller limit means fewer d's to iterate.
        # However, if no solution exists for limit=10, we try 20, etc.
        
        # Let's try from small to large.
        # If we find a solution with max denom M, we know the answer is <= M.
        # We can then try to find a solution with max denom < M.
        
        pass # The loop above is not quite right for optimization.
    
    # Better approach:
    # 1. Find any valid solution quickly (e.g., seed).
    current_solution = [8, 28, 1736]
    best_max_denom = 1736
    best_solution = current_solution
    
    # 2. Try to improve by searching for solutions with max denom < best_max_denom.
    # We can do a binary search on the max denominator?
    # Or just try a few specific limits below 1736.
    
    # Let's try limits: 100, 200, 500, 1000, 1500.
    # If we find a solution with limit L, we update best and try L//2, etc.
    
    def find_solution_with_max_limit(limit):
        # Returns a solution if found, else None.
        # Uses DFS with max_denom_limit = limit.
        # To speed up, we can also prune if the current sum of 1/d for d in range is not enough?
        # No, we are summing TO target, not FROM.
        
        # We need to be careful with recursion depth and time.
        
        def dfs_inner(rem, min_denom, current_denoms):
            if time.time() - start_time > max_time:
                return None
            if rem == 0:
                return list(current_denoms)
            if rem < 0:
                return None
                
            lower_bound = max(min_denom, (rem.denominator + rem.numerator - 1) // rem.numerator)
            if lower_bound > limit:
                return None
                
            # Optimization: if we have many terms, the sum might be too large?
            # No, we are constructing the sum.
            
            # Pruning: if rem is very small, we need large denominators.
            # If rem is very large, we need small denominators.
            
            # Iterate d
            for d in range(lower_bound, limit + 1):
                if time.time() - start_time > max_time:
                    return None
                next_rem = rem - Fraction(1, d)
                if next_rem < 0:
                    continue # Should not happen if lower_bound is correct, but safe.
                
                current_denoms.append(d)
                res = dfs_inner(next_rem, d + 1, current_denoms)
                current_denoms.pop()
                if res is not None:
                    return res
            return None
            
        return dfs_inner(target, 1, [])

    # Try to find a better solution
    # We know 1736 works.
    # Let's try 1000, 500, 200, 100.
    
    for limit in [1000, 500, 200, 100, 50]:
        if time.time() - start_time > max_time:
            break
        sol = find_solution_with_max_limit(limit)
        if sol:
            max_d = max(sol)
            if max_d < best_max_denom:
                best_max_denom = max_d
                best_solution = sol
                # If we found a solution, we can try to go even lower?
                # But we are iterating limits in decreasing order.
                # If we found one at 1000, we try 500 next.
                # If we find one at 500, we update.
                # This is good.
    
    # If we still have time, try to refine around the best found.
    # Or try a few more random limits?
    
    if best_solution is None:
        return [8, 28, 1736]
    
    return best_solution

# EVOLVE-BLOCK-END
