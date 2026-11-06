"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import log

def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    target = Fraction(5, 31)
    
    # The smallest possible denominator: 1/7 < 5/31 < 1/6, so denominators >= 7
    # We need to find the minimum M such that there exists a subset of {7,...,M}
    # whose reciprocals sum to exactly 5/31
    
    # Lower bound: need sum of some reciprocals from {7,...,M} to equal 5/31
    # The minimum M must be at least 7 (since 1/7 alone isn't enough, we need more terms)
    
    # Strategy: iterate M from 7 upward, for each M check if a valid subset exists
    # Use DFS with strong pruning based on suffix sums
    
    best_solution = None
    best_M = 999999999
    
    # Precompute: for a given M, the total sum of 1/k for k in [7, M]
    # If this is less than target, M is too small
    
    # We'll search M from 7 to some reasonable upper bound
    # 5/31 ≈ 0.1613, and sum_{k=7}^{M} 1/k ≈ ln(M/6) + gamma - sum_{k=1}^{6} 1/k
    # For M=8: 1/7+1/8 = 15/56 ≈ 0.2679 > 0.1613, so M=8 might work with 2 terms
    # But 1/7 + 1/8 = 15/56 ≠ 5/31
    # Let's just iterate
    
    for M in range(7, 200):
        if time.time() > deadline:
            break
        
        # Quick check: is the total available sum >= target?
        # sum_{k=7}^{M} 1/k
        total_avail = sum(Fraction(1, k) for k in range(7, M + 1))
        if total_avail < target:
            continue
        
        # Also check: can we form the target? 
        # We need a subset of {7,...,M} summing to 5/31
        # Use DFS with pruning
        
        # Precompute suffix sums: suf[i] = sum(1/k for k in range(i, M+1))
        # suf[M+1] = 0
        # suf[i] = 1/i + suf[i+1]
        suf = [Fraction(0)] * (M + 3)
        for i in range(M, 6, -1):
            suf[i] = suf[i + 1] + Fraction(1, i)
        
        found = [False]
        solution = []
        
        def dfs(remaining, start, depth):
            if found[0]:
                return
            if time.time() > deadline:
                found[0] = True  # abort
                return
            
            if remaining == 0:
                found[0] = True
                return
            
            if start > M:
                return
            
            # Pruning: max we can add from start..M
            if suf[start] < remaining:
                return
            
            # If even taking all remaining is not enough, prune
            # If 1/start > remaining, we can't use start, must skip
            if Fraction(1, start) > remaining:
                # Can't use start, try next
                dfs(remaining, start + 1, depth + 1)
                return
            
            # Option 1: don't use 'start'
            dfs(remaining, start + 1, depth + 1)
            if found[0]:
                return
            
            # Option 2: use 'start'
            solution.append(start)
            dfs(remaining - Fraction(1, start), start + 1, depth + 1)
            solution.pop()
        
        dfs(target, 7, 0)
        
        if found[0] and solution:
            # Verify the solution
            s = sum(Fraction(1, d) for d in solution)
            if s == target and len(set(solution)) == len(solution) and all(7 <= d <= M for d in solution):
                best_solution = solution[:]
                best_M = M
                break
        elif found[0]:
            # Timed out during search for this M
            break
    
    if best_solution is None:
        # Fallback: try a known solution or use the best from previous approach
        # 5/31 = 1/8 + 1/31 + 1/248? Let me check: 1/8 + 1/31 + 1/248
        # = 31/248 + 8/248 + 1/248 = 40/248 = 5/31. Yes!
        best_solution = [8, 31, 248]
    
    return best_solution
# EVOLVE-BLOCK-END
