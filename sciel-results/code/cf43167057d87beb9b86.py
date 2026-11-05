"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # A[i] = L / (i+1) for i in 0..n-1
    A = [L // (i + 1) for i in range(n)]
    
    # Split into two halves
    mid = n // 2
    A1 = A[:mid]
    A2 = A[mid:]
    
    # Generate all possible sums for first half
    sums1 = [0]
    for a in A1:
        new_sums = []
        for s in sums1:
            new_sums.append(s - a)
            new_sums.append(s)
            new_sums.append(s + a)
        sums1 = new_sums
        if time.time() > deadline:
            # Fallback to a simple heuristic if too slow
            return [0] * 28 + [1, -1]
    
    sums1.sort()
    
    # Generate all possible sums for second half and search
    best_N = None
    best_abs_N = float('inf')
    best_coeffs = None
    
    sums2 = [0]
    for a in A2:
        new_sums = []
        for s in sums2:
            new_sums.append(s - a)
            new_sums.append(s)
            new_sums.append(s + a)
        sums2 = new_sums
        if time.time() > deadline:
            break
    
    # For each sum in second half, find best match in first half
    for s2 in sums2:
        if time.time() > deadline:
            break
        
        # We want s1 + s2 to be close to 0, so s1 should be close to -s2
        target = -s2
        # Binary search for target in sums1
        # We want to check the insertion point and its neighbors
        idx = 0
        lo, hi = 0, len(sums1) - 1
        while lo <= hi:
            mid_idx = (lo + hi) // 2
            if sums1[mid_idx] < target:
                lo = mid_idx + 1
            else:
                hi = mid_idx - 1
        # lo is the index of the first element >= target
        # Check lo and lo-1
        
        candidates = []
        if lo < len(sums1):
            candidates.append(sums1[lo])
        if lo - 1 >= 0:
            candidates.append(sums1[lo - 1])
        # Also check lo+1 and lo-2 just in case of ties or close values
        if lo + 1 < len(sums1):
            candidates.append(sums1[lo + 1])
        if lo - 2 >= 0:
            candidates.append(sums1[lo - 2])
        
        for s1 in candidates:
            total_N = s1 + s2
            if total_N == 0:
                continue
            abs_total = abs(total_N)
            if abs_total < best_abs_N:
                best_abs_N = abs_total
                best_N = total_N
                # Reconstruct coefficients
                # We need to find the specific coefficients that produced s1 and s2
                # This is expensive. Instead, we can store the path during generation.
                # But storing paths for 3^15 items is memory heavy.
                # Alternative: We can just return the value if we don't need coefficients?
                # No, we must return coefficients.
                # Let's reconstruct by backtracking.
                # Actually, we can just find ANY combination that gives this sum.
                # Since we are in a tight loop, let's defer reconstruction.
                # We'll store the best s1, s2 and reconstruct at the end.
                pass
                
    if best_abs_N == float('inf'):
        # Fallback
        return [0] * 28 + [1, -1]
        
    # We need to reconstruct the coefficients.
    # This requires storing the paths or doing a second pass.
    # Given the time limit, let's try a simpler approach if the above is too complex.
    # The above approach finds the minimum |N|, but not the coefficients.
    # Let's modify the search to store the best s1 and s2.
    
    # Restart with coefficient tracking? No, too slow.
    # Let's assume we found the best value. Now find coefficients.
    # We can use a recursive search with pruning, or just store the best s1/s2.
    
    # Let's refine the search to track the best s1 and s2.
    best_s1 = None
    best_s2 = None
    
    # Re-run the second half loop to find the specific s1, s2
    for s2 in sums2:
        if time.time() > deadline:
            break
        target = -s2
        lo, hi = 0, len(sums1) - 1
        while lo <= hi:
            mid_idx = (lo + hi) // 2
            if sums1[mid_idx] < target:
                lo = mid_idx + 1
            else:
                hi = mid_idx - 1
        candidates = []
        if lo < len(sums1):
            candidates.append(sums1[lo])
        if lo - 1 >= 0:
            candidates.append(sums1[lo - 1])
        if lo + 1 < len(sums1):
            candidates.append(sums1[lo + 1])
        if lo - 2 >= 0:
            candidates.append(sums1[lo - 2])
        
        for s1 in candidates:
            total_N = s1 + s2
            if total_N == 0:
                continue
            abs_total = abs(total_N)
            if abs_total < best_abs_N:
                best_abs_N = abs_total
                best_s1 = s1
                best_s2 = s2

    if best_s1 is None or best_s2 is None:
        return [0] * 28 + [1, -1]
        
    # Reconstruct coefficients for best_s1 from A1
    def find_coeffs(A, target):
        n = len(A)
        if n == 0:
            return [] if target == 0 else None
        a = A[0]
        # target = c0*a + rest
        # rest = target - c0*a
        # rest must be achievable by A[1:]
        # Since we don't have a set of achievable sums for A[1:], we can't easily prune.
        # But we can just try all 3 possibilities for c0 and recurse.
        # This is 3^n, which is 3^15 = 14M. Might be slow if done naively.
        # But we only do it once.
        # To speed up, we can use a memoization or set of remaining sums?
        # Let's just do a simple DFS with pruning if possible.
        # Actually, we can generate the set of all sums for A[1:] once?
        # No, we already have sums1 which is all sums for A1.
        # We can check if (target - c0*a) is in sums(A[1:]).
        # But we don't have sums(A[1:]) stored.
        # Let's just do a simple backtracking.
        
        # Optimization: Use the fact that we know a solution exists.
        # We can use a greedy approach? No, not always works.
        # Let's use a set of reachable sums for the remaining elements.
        # Precompute reachable sums for each suffix?
        # reachable[i] = set of sums from A[i:]
        # This uses too much memory.
        
        # Let's just do a simple recursive search.
        # It might take a few seconds.
        
        def dfs(idx, remaining):
            if idx == n:
                return [] if remaining == 0 else None
            a = A[idx]
            for c in [-1, 0, 1]:
                res = dfs(idx + 1, remaining - c * a)
                if res is not None:
                    return [c] + res
            return None
        
        return dfs(0, target)
    
    coeffs1 = find_coeffs(A1, best_s1)
    coeffs2 = find_coeffs(A2, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
        
    coeffs = coeffs1 + coeffs2
    
    # Verify
    S = sum(c / (i + 1) for i, c in enumerate(coeffs))
    if S == 0:
        return [0] * 28 + [1, -1]
        
    return coeffs
# EVOLVE-BLOCK-END
