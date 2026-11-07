"""Solve: find 10 distinct positive integers <=1e6, each divisible by its divisor, minimizing range."""
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Compute LCM of all divisors for periodicity
    lcm_all = reduce(_lcm, divisors)
    # lcm_all = 32730
    
    # Deadline guard
    deadline = time.time() + 80.0
    
    best_result = None
    best_range = float('inf')
    
    # Strategy: For each possible window width W (starting from small),
    # and for each offset within the LCM period, check if we can assign
    # distinct multiples in [L, L+W] for all divisors.
    #
    # Key insight: The pattern of which multiples are available repeats every lcm_all.
    # So we only need to check L modulo lcm_all.
    
    # For a given window [L, L+W], divisor d has multiples at positions
    # k*d where ceil(L/d) <= k <= floor((L+W)/d).
    # The number of available multiples for d is floor((L+W)/d) - ceil(L/d) + 1.
    #
    # We need to assign distinct values. This is a bipartite matching problem.
    # But since n=10, we can use backtracking.
    
    def can_assign(L, W):
        """Check if we can assign distinct multiples in [L, L+W] for each divisor."""
        # For each divisor, compute the list of available multiples
        avail = []
        for d in divisors:
            k_min = (L + d - 1) // d
            k_max = (L + W) // d
            if k_max < k_min:
                return None
            multiples = [k * d for k in range(k_min, k_max + 1)]
            avail.append(multiples)
        
        # Backtracking: assign values to divisors
        # Order divisors by number of available multiples (most constrained first)
        order = sorted(range(n), key=lambda i: len(avail[i]))
        
        used = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for v in avail[i]:
                if v not in used:
                    used.add(v)
                    if backtrack(idx + 1):
                        return True
                    used.remove(v)
            return False
        
        if backtrack(0):
            # Reconstruct the assignment
            result = [0] * n
            used.clear()
            
            def backtrack2(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in avail[i]:
                    if v not in used:
                        used.add(v)
                        result[i] = v
                        if backtrack2(idx + 1):
                            return True
                        used.remove(v)
                        result[i] = 0
                return False
            
            backtrack2(0)
            return result
        return None
    
    # Search over window widths and offsets
    # The minimum possible range: we need 10 distinct values, so range >= 9.
    # But divisibility constraints may force larger ranges.
    
    # For each window width W from 10 upward:
    #   For each offset r in [0, lcm_all-1]:
    #     L = r (or r + t*lcm_all for some t, but the pattern repeats)
    #     Actually, for a given W and r, the set of available multiples depends on L mod d for each d.
    #     Since lcm_all is a multiple of all d, the pattern of available multiples
    #     in [L, L+W] depends on L mod lcm_all.
    
    # But W might be larger than lcm_all? Unlikely for small W.
    # For W < lcm_all, the pattern in [L, L+W] is determined by L mod lcm_all.
    
    # Let's iterate W from 10 to some limit, and for each W, check offsets.
    # To keep it fast, we can check a subset of offsets or use a smarter approach.
    
    # Actually, let's just iterate L over a range and check.
    # Since lcm_all = 32730, and we want small W, let's try:
    # For W in range(10, 500):
    #   For L in range(0, lcm_all):  # but this is 32730 * 490 = too many
    #
    # Better: for each W, the condition depends on L mod d for each d.
    # The combined period is lcm_all. So we check L in [0, lcm_all).
    # That's 32730 checks per W, and W goes up to maybe 200-300.
    # 32730 * 300 = ~10 million checks, each with backtracking on 10 items.
    # Might be too slow. Let's optimize.
    
    # Optimization: precompute for each divisor d and each residue r mod d,
    # the number of multiples in [r, r+W] ... no, that's not quite right.
    
    # Alternative: for each W, iterate L from 0 to lcm_all-1, but skip quickly
    # if any divisor has 0 available multiples.
    
    # Let's try a different approach: iterate L in steps, but use the LCM structure.
    
    # Actually, let's just do a direct search with a time limit.
    # Try W from 10 upward, and for each W, try L values.
    
    for W in range(10, 1000):
        if time.time() > deadline:
            break
        
        # For this W, check L values. The pattern repeats every lcm_all.
        # So check L in [0, lcm_all).
        # But we can be smarter: for divisor d, the number of multiples in [L, L+W]
        # is either floor(W/d) or floor(W/d) + 1, depending on L mod d.
        # We need at least 1 multiple for each divisor.
        
        # Quick check: for each d, we need W >= d-1 (at minimum, if L is just after a multiple)
        # Actually, if W >= d-1, then any interval of length W contains at least one multiple of d.
        # If W < d-1, then some intervals of length W contain no multiple of d.
        
        # So we need W >= max(d) - 1 = 39 - 1 = 38 for guaranteed coverage.
        # But we might get lucky with smaller W for specific L.
        
        # Let's just check L values. To speed up, check L in a smarter order.
        
        # For each L in [0, lcm_all), check if all divisors have at least one multiple.
        # Use early termination.
        
        for L in range(0, lcm_all):
            if time.time() > deadline:
                break
            
            # Quick check: does each divisor have at least one multiple in [L, L+W]?
            feasible = True
            for d in divisors:
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                if k_max < k_min:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Try to assign
            result = can_assign(L, W)
            if result is not None:
                # Check all values are <= max_val
                if max(result) <= max_val:
                    if W < best_range:
                        best_range = W
                        best_result = result[:]
                    # Found a solution for this W, no need to check larger L for same W
                    break
        
        # If we found a solution for this W, we can stop (since we're increasing W)
        if best_result is not None and best_range == W:
            break
    
    if best_result is None:
        # Fallback: just use the divisors themselves (range = 39-6 = 33)
        best_result = divisors[:]
    
    # Ensure types are correct
    return [int(x) for x in best_result]
