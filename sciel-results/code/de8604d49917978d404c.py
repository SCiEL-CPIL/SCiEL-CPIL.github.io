"""Rewritten with branch-and-bound MWIS solver."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction
from functools import lru_cache

def construct():
    deadline = time.time() + 55  # 55s safety margin before 60s kill
    
    # Build universe and conflict graph
    # Two numbers conflict if lcm(a,b) <= 256
    n_max = 256
    universe = list(range(2, n_max + 1))
    N = len(universe)
    idx = {v: i for i, v in enumerate(universe)}
    
    # Weights: 1/v
    weights = [Fraction(1, v) for v in universe]
    
    # Conflict graph: edge (i,j) if lcm(universe[i], universe[j]) <= 256
    # Build adjacency as bitmasks (Python ints work as arbitrary-precision bitsets)
    # For N=255, we need 255-bit integers
    conflict = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if lcm(universe[i], universe[j]) <= n_max:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Greedy seed solution
    def greedy_seed():
        selected = []
        for v in universe:
            if all(lcm(v, y) > n_max for y in selected):
                selected.append(v)
        return selected
    
    seed_selected = greedy_seed()
    seed_set = set(seed_selected)
    seed_val = sum(Fraction(1, v) for v in seed_selected)
    
    best_val = seed_val
    best_set = set(seed_set)
    
    # Branch and bound for MWIS
    # Order vertices by descending weight (smaller numbers first)
    # Process in order: for each vertex, decide include or exclude
    # Upper bound: current_sum + sum of weights of all remaining candidates that could be added
    
    # Precompute: for each vertex, which later vertices conflict with it
    # We'll process vertices in order 0..N-1 (which is 2..256, i.e., ascending value = descending weight)
    
    # Key optimization: precompute for each vertex the mask of conflicts with later vertices
    # and the sum of weights of non-conflicting later vertices (as upper bound contribution)
    
    # Actually, a better bound: at each step, the remaining candidates are those not yet processed
    # and not conflicting with already-selected vertices. The upper bound is current + sum of all
    # remaining candidate weights.
    
    # Let's use a recursive approach with memoization on (position, available_mask) 
    # but that's too large. Instead, use branch and bound with good pruning.
    
    # Sort by descending weight (already in order since universe is ascending)
    # Vertex i has weight 1/universe[i], so vertex 0 (value 2) has highest weight.
    
    # Precompute suffix sums of weights for upper bound
    suffix_weight = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_weight[i] = suffix_weight[i + 1] + weights[i]
    
    # For tighter bound: precompute for each vertex i, the mask of vertices j > i that conflict with i
    later_conflict = [0] * N
    for i in range(N):
        mask = conflict[i] & ((1 << N) - 1)
        # Keep only bits j > i
        later_conflict[i] = mask & (~((1 << (i + 1)) - 1))
    
    # Branch and bound
    # State: (pos, available_mask, current_sum)
    # available_mask: bitmask of vertices from pos onward that are still available (not conflicting with selected)
    
    # To make this tractable, we'll use iterative deepening or just careful branching.
    # The key pruning: if current_sum + suffix_weight[pos] <= best_val, prune.
    # Even tighter: current_sum + sum of weights of available vertices from pos onward <= best_val.
    
    # Let's precompute available weight sums... actually we need to compute dynamically.
    # For the bound, use: current_sum + sum(w[i] for i in available_mask & suffix_mask[pos])
    # This is expensive to compute each time. Use the looser bound first: current_sum + suffix_weight[pos]
    
    # Better approach: use the fact that if we include vertex i, we remove its conflicts from available.
    # If we exclude vertex i, we just move to i+1.
    
    # Let's implement with a stack-based DFS to avoid recursion depth issues.
    
    # Actually, let's use a smarter approach:
    # 1. Start with all vertices available
    # 2. At each step, pick the highest-weight available vertex
    # 3. Branch: include it (remove it and its conflicts) or exclude it (remove just it)
    # 4. Bound: current + sum of all remaining available weights
    
    # Precompute weight lookup
    w_list = [float(w) for w in weights]  # use floats for speed in bounds, Fraction for exact comparison
    
    def solve():
        nonlocal best_val, best_set, deadline
        
        # Use a priority-based approach: always branch on the highest weight available vertex
        # This gives better bounds earlier
        
        # We'll use a simple recursive DFS with the following strategy:
        # - available: set of available vertex indices (use a frozenset or sorted list)
        # - current: current sum (Fraction)
        # - selected: set of selected vertices
        
        # For efficiency, represent available as a sorted list and use the first element (highest weight)
        
        # Initial state: all vertices available
        initial_available = list(range(N))
        
        # Use a stack for iterative DFS
        # Each entry: (available_list, current_frac, selected_set)
        # But this uses a lot of memory. Let's use recursion with a depth limit.
        
        # Actually, let's try a different approach: 
        # Since the graph has structure (conflicts are between small numbers mostly),
        # let's identify the "core" of conflicts and handle them separately.
        
        # Numbers >= 128: lcm(a,b) >= max(a,b) >= 128. lcm(a,b) <= 256 only if both are small.
        # Actually lcm(a,b) >= max(a,b), so if a >= 129, lcm(a,b) >= 129. 
        # lcm(a,b) <= 256 requires max(a,b) <= 256 AND lcm <= 256.
        # If a > 128, then b must divide a*256/a... let me think.
        # lcm(a,b) = a*b/gcd(a,b) <= 256. If a >= 129, then b/gcd(a,b) <= 256/a < 2, so b/gcd(a,b) = 1, meaning b divides a.
        # So for a >= 129, b must be a divisor of a (and b < a for distinct).
        
        # This means large numbers only conflict with their divisors!
        
        # Let's separate: small numbers (2..128) and large numbers (129..256)
        # Large numbers only conflict with their proper divisors (which are <= 128).
        # Two large numbers a,b >= 129: lcm(a,b) >= max(a,b) >= 129. 
        # lcm(a,b) <= 256? Only if b divides a or a divides b (since both > 128, gcd < min(a,b), so lcm > max).
        # Wait: if a,b >= 129 and a != b, can lcm(a,b) <= 256?
        # lcm(a,b) = a*b/gcd(a,b). Since a,b >= 129, a*b >= 129*130 = 16770. 
        # gcd(a,b) would need to be >= 16770/256 ≈ 65.5, so gcd >= 66.
        # But gcd(a,b) <= min(a,b)/2 if a != b (since if gcd = min, then min divides max).
        # If a divides b, then lcm = b <= 256, and a >= 129, so b is a multiple of a with b <= 256.
        # So b = 2a, requiring a <= 128. Contradiction since a >= 129.
        # So two large numbers never conflict! They're all mutually compatible.
        
        # This is a huge simplification. Let me verify:
        # a=129, b=258 > 256. No.
        # a=130, b=260 > 256. No.
        # In general, if a >= 129 and b is a multiple of a, b >= 2a >= 258 > 256.
        # So indeed, no two numbers >= 129 conflict with each other.
        
        # So the structure is:
        # - Small-small conflicts: between numbers in [2, 128]
        # - Small-large conflicts: a small number s conflicts with large number L if L is a multiple of s (and L/s is integer, L <= 256)
        #   Wait, lcm(s, L) = L if s divides L. So s conflicts with L iff s | L.
        # - Large-large: no conflicts.
        
        # This means: once we decide which small numbers to include, ALL large numbers 
        # that don't conflict with any included small number can be included for free!
        
        # So the problem reduces to:
        # 1. Choose a subset S of [2, 128] such that lcm(a,b) > 256 for all distinct a,b in S.
        # 2. Include all L in [129, 256] such that no s in S divides L.
        # 3. Maximize sum(1/s for s in S) + sum(1/L for included L).
        
        # This is much more tractable! The small part has only 127 vertices.
        
        # Let's rebuild with this structure.
        small = list(range(2, 129))  # 2..128
        large = list(range(129, 257))  # 129..256
        n_small = len(small)
        n_large = len(large)
        
        small_idx = {v: i for i, v in enumerate(small)}
        large_idx = {v: i for i, v in enumerate(large)}
        
        # Small weights
        sw = [Fraction(1, v) for v in small]
        # Large weights
        lw = [Fraction(1, v) for v in large]
        
        # Small-small conflict: lcm(small[i], small[j]) <= 256
        ss_conflict = [0] * n_small
        for i in range(n_small):
            for j in range(i + 1, n_small):
                if lcm(small[i], small[j]) <= 256:
                    ss_conflict[i] |= (1 << j)
                    ss_conflict[j] |= (1 << i)
        
        # Small-large conflict: small[i] divides large[j]
        # For each small number s, which large numbers L are multiples of s?
        sl_conflict = [0] * n_small  # bitmask of large indices for each small
        for i, s in enumerate(small):
            for j, L in enumerate(large):
                if L % s == 0:
                    sl_conflict[i] |= (1 << j)
        
        # Now the problem:
        # Choose S (bitmask over n_small bits) that is an independent set in ss_conflict.
        # For each such S, the included large numbers are those not in union of sl_conflict[i] for i in S.
        # Score = sum(sw[i] for i in S) + sum(lw[j] for j not in union_sl(S))
        
        # Let's precompute for each small vertex i:
        # - ss_conflict[i]: which small vertices conflict
        # - sl_conflict[i]: which large vertices are excluded
        
        # Upper bound for a partial selection:
        # current_small_sum + sum of remaining small weights + sum of all large weights
        # (loose, but fast)
        
        # Tighter: current_small_sum + sum of remaining small weights that don't conflict with selected
        # + sum of large weights for large numbers not excluded by selected OR remaining smalls
        
        # Let's use: current_small_sum + suffix_small[pos] + total_large_weight
        # where suffix_small[pos] = sum of sw[pos:]
        
        # Even better bound considering large exclusions:
        # For the large part, if we've selected some smalls, the excluded large set is fixed.
        # Remaining smalls can only exclude MORE large numbers, never fewer.
        # So upper bound on large contribution = sum of lw[j] for j not excluded by current selection.
        # This is a valid upper bound (optimistic, since remaining smalls might exclude more).
        
        # Let's compute this bound.
        
        # Precompute suffix small weights
        suffix_sw = [Fraction(0)] * (n_small + 1)
        for i in range(n_small - 1, -1, -1):
            suffix_sw[i] = suffix_sw[i + 1] + sw[i]
        
        total_lw = sum(lw)
        
        # For fast computation of "sum of lw[j] for j not in excluded_mask":
        # Precompute prefix sums? Or just compute on the fly.
        # n_large = 128, so iterating is fine.
        
        # Branch and bound over small vertices
        # Process in order 0..n_small-1 (ascending value = descending weight)
        
        best_score = None
        best_small_mask = 0
        best_large_excluded = 0
        
        # Initial greedy for a good starting bound
        def greedy_small():
            selected = []
            for i, s in enumerate(small):
                if all(lcm(s, y) > 256 for y in selected):
                    selected.append(s)
            return selected
        
        greedy_sel = greedy_small()
        greedy_mask = 0
        greedy_excluded = 0
        for s in greedy_sel:
            i = small_idx[s]
            greedy_mask |= (1 << i)
            greedy_excluded |= sl_conflict[i]
        
        greedy_score = sum(Fraction(1, s) for s in greedy_sel)
        for j, L in enumerate(large):
            if not (greedy_excluded >> j) & 1:
                greedy_score += Fraction(1, L)
        
        best_score = greedy_score
        best_small_mask = greedy_mask
        best_large_excluded = greedy_excluded
        
        # Also try: include ALL large numbers (exclude no smalls)
        all_large_score = total_lw
        if all_large_score > best_score:
            best_score = all_large_score
            best_small_mask = 0
            best_large_excluded = 0
        
        # Branch and bound
        # We'll use a recursive function with explicit time checking
        
        def compute_large_score(excluded_mask):
            """Compute sum of 1/L for L in large not in excluded_mask."""
            s = Fraction(0)
            for j in range(n_large):
                if not (excluded_mask >> j) & 1:
                    s += lw[j]
            return s
        
        # For speed, use float for bounds, Fraction for exact comparison
        sw_float = [float(w) for w in sw]
        lw_float = [float(w) for w in lw]
        suffix_sw_float = [0.0] * (n_small + 1)
        for i in range(n_small - 1, -1, -1):
            suffix_sw_float[i] = suffix_sw_float[i + 1] + sw_float[i]
        total_lw_float = sum(lw_float)
        
        def solve_bnb():
            nonlocal best_score, best_small_mask, best_large_excluded, deadline
            
            # Recursive B&B
            # pos: next small vertex to consider
            # selected_mask: bitmask of selected small vertices so far
            # excluded_large: bitmask of large vertices excluded so far
            # current_score: Fraction, current small sum
            
            # Use iterative approach with explicit stack to control time
            # Stack entries: (pos, selected_mask, excluded_large, current_score)
            
            # Actually, recursion with time check is simpler. Let's use sys.setrecursionlimit.
            import sys
            sys.setrecursionlimit(10000)
            
            def dfs(pos, selected_mask, excluded_large, current_score):
                nonlocal best_score, best_small_mask, best_large_excluded, deadline
                
                if time.time() > deadline:
                    return
                
                if pos == n_small:
                    # All small vertices processed
                    large_score = compute_large_score(excluded_large)
                    total = current_score + large_score
                    if total > best_score:
                        best_score = total
                        best_small_mask = selected_mask
                        best_large_excluded = excluded_large
                    return
                
                # Upper bound: current_score + suffix_sw[pos] + (large score with current exclusions)
                # The large score with current exclusions is an upper bound (remaining smalls can only add exclusions)
                # But computing it exactly each time is expensive. Use float approximation.
                
                # Quick float bound:
                # lb_large = sum of lw_float[j] for j not in excluded_large
                # This is expensive. Use a looser bound: total_lw_float
                # Even looser: current_score_float + suffix_sw_float[pos] + total_lw_float
                
                current_float = float(current_score)
                if current_float + suffix_sw_float[pos] + total_lw_float <= float(best_score) + 1e-15:
                    return  # Pruned
                
                # Option 1: Include vertex pos
                # Check if it conflicts with any already selected
                if (selected_mask & ss_conflict[pos]) == 0:
                    new_mask = selected_mask | (1 << pos)
                    new_excluded = excluded_large | sl_conflict[pos]
                    new_score = current_score + sw[pos]
                    dfs(pos + 1, new_mask, new_excluded, new_score)
                
                # Option 2: Exclude vertex pos
                dfs(pos + 1, selected_mask, excluded_large, current_score)
            
            dfs(0, 0, 0, Fraction(0))
        
        solve_bnb()
        
        # Build final set
        result = []
        for i in range(n_small):
            if (best_small_mask >> i) & 1:
                result.append(small[i])
        for j in range(n_large):
            if not (best_large_excluded >> j) & 1:
                result.append(large[j])
        
        # Verify
        if not result:
            return [2]
        
        # Verify constraint
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if lcm(result[i], result[j]) <= 256:
                    # Shouldn't happen, but fallback to greedy
                    return seed_selected
        
        return result
    
    return solve()
# EVOLVE-BLOCK-END
