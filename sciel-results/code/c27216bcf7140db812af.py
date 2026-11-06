"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55.0
    target = Fraction(5, 31)
    
    # Fallback
    best_solution = [8, 31, 248]
    best_max = 248

    # Iterate possible maximum denominators M
    # Lower bound: 1/(5/31) = 6.2, so min denominator is 7.
    # Max denominator M must be at least 7.
    # We search for the smallest M such that a valid expansion exists.
    
    for M in range(7, 100):
        if time.time() > deadline:
            break
            
        # Candidates are denominators in [7, M]
        # We need distinct denominators.
        # We perform a DFS to see if any subset of {7, ..., M} sums to 5/31.
        # To make DFS efficient, we use the list of candidates sorted descending? 
        # Or ascending?
        # Let's use ascending order d_7, ..., d_M.
        # We pick or don't pick each.
        # Pruning:
        # 1. Current sum + max_possible_remaining < target -> prune
        # 2. Current sum > target -> prune
        
        candidates = list(range(7, M + 1))
        
        # Precompute suffix sums of reciprocals for pruning
        # suffix_sum[i] = sum(1/d for d in candidates[i:])
        n = len(candidates)
        suffix_sum = [Fraction(0, 1)] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, candidates[i])
            
        # Also precompute prefix? No, we are building from start.
        # We need to check if remaining target can be formed by subset of candidates[i:].
        # Max sum available is suffix_sum[i].
        # Min sum is 0.
        # Also, we can't use fewer than k terms if target > k * max_reciprocal? 
        # Simplest bound: if remaining_target > suffix_sum[i], prune.
        
        found = False
        solution = []
        
        def dfs(idx, current_sum, path):
            nonlocal found, solution
            if found:
                return
            if time.time() > deadline:
                return
                
            if current_sum == target:
                if len(path) > 0:
                    found = True
                    solution = path[:]
                return
                
            if idx >= n:
                return
                
            # Pruning:
            # 1. If current_sum > target, prune (since all terms positive)
            if current_sum > target:
                return
            
            # 2. If current_sum + suffix_sum[idx] < target, prune
            if current_sum + suffix_sum[idx] < target:
                return
                
            # Option 1: Include candidates[idx]
            d = candidates[idx]
            new_sum = current_sum + Fraction(1, d)
            # Additional pruning: if new_sum > target, we can't include it (and since candidates are increasing, 
            # including larger ones will also exceed? No, we are picking from a set, order doesn't matter for sum, 
            # but we are iterating in order. If we skip, we move to next. If we take, we move to next.
            # If new_sum > target, we cannot take this d. We must skip it.
            if new_sum <= target:
                path.append(d)
                dfs(idx + 1, new_sum, path)
                path.pop()
                
            # Option 2: Exclude candidates[idx]
            # We can only exclude if it's possible to reach target with remaining elements.
            # This is covered by the recursive call.
            # But we can prune if even taking ALL remaining elements (except current) is not enough?
            # That is: current_sum + suffix_sum[idx+1] < target.
            if current_sum + suffix_sum[idx + 1] >= target:
                dfs(idx + 1, current_sum, path)
            else:
                # If we skip this one, we definitely can't reach target with the rest.
                # So we don't need to recurse for the "skip" case.
                pass

        dfs(0, Fraction(0, 1), [])
        
        if found:
            # We found a solution with max denominator <= M
            # Since we iterate M from low to high, this is the optimal M.
            # The solution found has max denominator <= M.
            # It might be less than M if M was not used, but we started from 7.
            # The actual max denominator in 'solution' is max(solution).
            # We return the solution.
            # Ensure distinctness (DFS ensures this by structure) and range (DFS ensures this).
            return solution

    # If no solution found in range, return fallback
    return best_solution
# EVOLVE-BLOCK-END
