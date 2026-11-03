import time
import random
import math
import numpy as np
from collections import defaultdict
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Lower bound estimation
    # Total distinct differences needed: n_rows * k * (k+1) / 2
    # Max difference is M. So M >= n_rows * k * (k+1) / 2
    lb = (n_rows * k * (k + 1)) // 2
    if lb < 1: lb = 1
    
    # 2. Generate candidate rows
    # We will generate rows with scope M and check if they are valid (distinct diffs within row)
    # and record their diff bitmask.
    
    def gen_random_row(M, rng):
        """Generate a random valid row with scope <= M."""
        row = [0]
        x = 0
        used_diffs = set()
        
        # Precompute valid next steps? No, just try.
        # To make it faster, we can precompute for a given M and k?
        # But M varies.
        
        for _ in range(k):
            found = False
            # Try random candidates in range (x, M]
            # Bias towards smaller gaps to keep scope low? Or uniform?
            # Uniform is fine, we filter by scope later.
            
            # Optimization: only try t such that t - x is small?
            # No, just random.
            
            candidates = list(range(x + 1, M + 1))
            if not candidates:
                return None
            
            rng.shuffle(candidates)
            
            # To speed up, limit shuffle size if M is large?
            # M is around 100-150. Shuffling 150 items is fast.
            
            for t in candidates:
                # Check if t - a is in used_diffs for all a in row
                ok = True
                for a in row:
                    d = t - a
                    if d in used_diffs:
                        ok = False
                        break
                if ok:
                    row.append(t)
                    for a in row[:-1]:
                        used_diffs.add(t - a)
                    x = t
                    found = True
                    break
            
            if not found:
                return None
        
        return row

    def get_diff_mask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    # Strategy:
    # We want to find the smallest M such that there exist n_rows rows with scope <= M
    # and disjoint diff masks.
    
    # We will iterate M from lb upwards.
    # For each M, generate a pool of candidates.
    # Then solve MIS.
    
    # To make this efficient, we should generate candidates for the *current best M* found so far,
    # and try to improve it.
    
    # Let's start with a greedy estimate to get an upper bound.
    
    # Greedy construction
    used_mask = 0
    greedy_rows = []
    M_greedy = 0
    rng_greedy = random.Random(123)
    
    # Simple greedy: pick smallest possible next element
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_mask = 0
            for a in row:
                new_mask |= (1 << (x - a))
            if (used_mask & new_mask) == 0:
                row.append(x)
                used_mask |= new_mask
                M_greedy = max(M_greedy, x)
        greedy_rows.append(row)
    
    best_scope = M_greedy
    best_rows = greedy_rows
    
    # Now, try to improve using MIS on candidate pools for M < best_scope
    
    # We'll try M values from lb to best_scope - 1
    # But generating a pool for each M is expensive.
    # Instead, generate a large pool for M = best_scope - 1, then best_scope - 2, etc.
    # Or, generate a pool for M = best_scope and try to find a subset with smaller max scope?
    # No, the scope is a property of the row.
    
    # Let's generate candidates for a range of M values.
    # Actually, it's better to fix M and try to find n_rows disjoint rows with scope <= M.
    
    # We will try M from lb to best_scope - 1.
    # For each M, generate N candidates.
    # If we find a solution, update best_scope = M and break (or continue to find even smaller? No, M is increasing).
    # Wait, if we find a solution for M, we should check if we can find one for M-1?
    # Yes, so we should iterate M from lb upwards.
    
    # However, generating candidates for small M might fail often.
    # Let's start from the greedy upper bound and go down?
    # No, "minimal scope" means we want the smallest M.
    # So iterate M from lb to best_scope - 1.
    
    # To save time, we can generate a pool of candidates for M = best_scope - 1.
    # If we can't find a solution, try M = best_scope - 2, etc.
    
    # But generating a pool takes time.
    # Let's generate a pool for M = best_scope - 1.
    
    # Actually, a smarter approach:
    # Generate a large pool of candidates for M = best_scope.
    # Many of these will have scope < best_scope.
    # Filter them by scope.
    # For each M from lb to best_scope - 1:
    #   Filter pool to rows with scope <= M.
    #   Try to find MIS of size n_rows.
    
    # This reuses the pool.
    
    N_CANDIDATES = 20000
    rng_pool = random.Random(42)
    
    candidates = []
    seen_masks = set()
    
    # Generate candidates with scope <= best_scope
    # We can bias the generation towards smaller scopes?
    # Just generate with M = best_scope.
    
    for _ in range(N_CANDIDATES):
        if time.time() > deadline - 10:
            break
        row = gen_random_row(best_scope, rng_pool)
        if row is None:
            continue
        mask = get_diff_mask(row)
        if mask in seen_masks:
            continue
        seen_masks.add(mask)
        scope = row[-1]
        candidates.append((scope, mask, row))
    
    if not candidates:
        return best_rows
    
    # Sort by scope
    candidates.sort(key=lambda x: x[0])
    
    # For each M from lb to best_scope - 1:
    #   Get candidates with scope <= M
    #   Try to find n_rows disjoint rows
    
    def solve_mis(cands, n_needed, deadline):
        """
        cands: list of (scope, mask, row)
        Find n_needed rows with disjoint masks.
        """
        if not cands:
            return None
        
        # Greedy + Local Search
        # Greedy: pick row with fewest conflicts? Or smallest scope?
        # Let's try greedy: pick random, then try to add more.
        
        best_solution = None
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            selected = []
            used_mask = 0
            remaining = cands[:]
            rng_local = random.Random(attempt)
            rng_local.shuffle(remaining)
            
            # Greedy selection
            for scope, mask, row in remaining:
                if len(selected) == n_needed:
                    break
                if (used_mask & mask) == 0:
                    selected.append((scope, mask, row))
                    used_mask |= mask
            
            if len(selected) == n_needed:
                # Check if this is better (smaller max scope)
                max_s = max(s for s, m, r in selected)
                if best_solution is None or max_s < best_solution[0]:
                    best_solution = (max_s, [r for s, m, r in selected])
                    if max_s == lb: # Can't do better
                        return best_solution
            
            # Local search: try to swap
            # Pick a selected row, remove it, try to replace with a different one
            for _ in range(5):
                if time.time() > deadline:
                    break
                if not selected:
                    break
                i = rng_local.randint(0, len(selected) - 1)
                old_scope, old_mask, old_row = selected[i]
                used_mask_without = 0
                for j, (s, m, r) in enumerate(selected):
                    if j != i:
                        used_mask_without |= m
                
                # Find a replacement
                replaced = False
                for scope, mask, row in cands:
                    if row == old_row:
                        continue
                    if (used_mask_without & mask) == 0:
                        # Swap
                        selected[i] = (scope, mask, row)
                        max_s = max(s for s, m, r in selected)
                        if best_solution is None or max_s < best_solution[0]:
                            best_solution = (max_s, [r for s, m, r in selected])
                            if max_s == lb:
                                return best_solution
                        replaced = True
                        break
                
                if not replaced:
                    # Try to add a new one? No, size is fixed.
                    pass
        
        return best_solution
    
    # Iterate M
    # We can binary search M? No, the property is not monotonic in a simple way for the *existence* of a solution?
    # Actually, if a solution exists for M, it exists for M+1. So it is monotonic.
    # So we can binary search M!
    
    lo = lb
    hi = best_scope - 1
    found_M = best_scope
    
    # But generating candidates for a specific M is the bottleneck.
    # Our pool is for M = best_scope.
    # If we binary search, we need to filter the pool.
    # If the pool is large enough, this works.
    
    # Let's just iterate M from lb to best_scope - 1, but stop early if we find a solution.
    # Since we want the *smallest* M, we iterate upwards.
    
    # Wait, if we iterate upwards, we might waste time on small M where no solution exists.
    # But the pool is fixed. Filtering is O(N).
    # MIS is the expensive part.
    
    # Let's try a few M values.
    # Start from lb.
    
    # To be safe, let's just try M = best_scope - 1 first.
    # If we find a solution, try M = best_scope - 2, etc.
    # This is "downhill" from the greedy bound.
    
    current_M = best_scope - 1
    while current_M >= lb and time.time() < deadline - 5:
        # Filter candidates
        valid_cands = [c for c in candidates if c[0] <= current_M]
        
        if len(valid_cands) < n_rows:
            current_M -= 1
            continue
        
        sol = solve_mis(valid_cands, n_rows, deadline - 2)
        
        if sol is not None:
            max_s, rows = sol
            if max_s < best_scope:
                best_scope = max_s
                best_rows = rows
                # Try to go lower
                # Update pool? No, just continue with smaller M.
                # But the pool might not have enough candidates for smaller M.
                # We can generate more candidates for the new best_scope?
                # Let's generate more candidates for M = best_scope.
                # Actually, let's just continue the loop.
                
                # If we found a solution, we can try to find a better one with the same M?
                # No, M is the bound. We want smaller M.
                
                # Let's try to generate more candidates for M = best_scope
                # to improve the pool for lower M searches.
                if time.time() < deadline - 10:
                    for _ in range(5000):
                        if time.time() > deadline - 10:
                            break
                        row = gen_random_row(best_scope, rng_pool)
                        if row is None:
                            continue
                        mask = get_diff_mask(row)
                        if mask in seen_masks:
                            continue
                        seen_masks.add(mask)
                        scope = row[-1]
                        candidates.append((scope, mask, row))
                    candidates.sort(key=lambda x: x[0])
                    
            current_M -= 1
        else:
            # No solution for this M
            current_M -= 1
    
    # Final check
    if _valid(best_rows, k):
        return best_rows
    
    # Fallback
    return greedy_rows

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

# Keep the original helper functions if needed, but the new construct is self-contained.
# The prompt asks to keep function names and signatures of the seed program.
# The seed program had: construct, _valid, _scope, _fast_greedy, _gen_random_row, _sa_vectorized, _dfs_row_minimize, _knapsack_packing, _branch_and_bound_search.
# I will define _scope and _fast_greedy for completeness, but they are not used in the new construct.

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _fast_greedy(k, n_rows, rng, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _gen_random_row(max_scope, k, rng, existing_diffs=None):
    return None # Placeholder

def _sa_vectorized(rows, k, deadline, rng, T_init=10.0, cooling=0.9997, min_T=0.05):
    return 0, rows # Placeholder

def _dfs_row_minimize(row, k, used_diffs, max_scope, deadline):
    return None # Placeholder

def _knapsack_packing(n_rows, k, deadline, rng):
    return None, None # Placeholder

def _branch_and_bound_search(n_rows, k, deadline, rng, initial_rows=None):
    return 0, None # Placeholder

