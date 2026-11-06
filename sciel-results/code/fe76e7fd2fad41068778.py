import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Candidates 2..256
    all_candidates = list(range(2, 257))
    
    # Split into Core (2..16) and Rest (17..256)
    core_candidates = list(range(2, 17))
    rest_candidates = list(range(17, 257))
    
    n_core = len(core_candidates)
    n_rest = len(rest_candidates)
    
    # Map rest value to index in rest_candidates
    rest_val_to_idx = {v: i for i, v in enumerate(rest_candidates)}
    
    # Precompute conflicts for Rest
    # rest_conflicts[i] is a set of indices j such that lcm(rest[i], rest[j]) <= 256
    rest_conflicts = [set() for _ in range(n_rest)]
    for i in range(n_rest):
        for j in range(i + 1, n_rest):
            a = rest_candidates[i]
            b = rest_candidates[j]
            g = gcd(a, b)
            if (a // g) * b <= 256:
                rest_conflicts[i].add(j)
                rest_conflicts[j].add(i)
                
    # Precompute core conflicts
    # core_conflicts[i] is a set of indices j
    core_conflicts = [set() for _ in range(n_core)]
    for i in range(n_core):
        for j in range(i + 1, n_core):
            a = core_candidates[i]
            b = core_candidates[j]
            g = gcd(a, b)
            if (a // g) * b <= 256:
                core_conflicts[i].add(j)
                core_conflicts[j].add(i)
                
    # Precompute core-rest compatibility
    # core_rest_conflicts[c_idx][r_idx] is True if conflict
    # Or better: for each core index c, list of rest indices that conflict with it
    core_conflicting_rest = [set() for _ in range(n_core)]
    for c_idx, c_val in enumerate(core_candidates):
        for r_idx, r_val in enumerate(rest_candidates):
            g = gcd(c_val, r_val)
            if (c_val // g) * r_val <= 256:
                core_conflicting_rest[c_idx].add(r_idx)
                
    best_global_sum = 0.0
    best_global_set = []
    
    # Enumerate all valid subsets of Core
    # Use recursion or bitmask
    # n_core = 15, so 2^15 = 32768 subsets.
    
    def is_valid_core_subset(mask):
        # Check if mask represents a valid independent set in core graph
        # Iterate bits
        for i in range(n_core):
            if mask & (1 << i):
                # Check against all other bits in mask
                # If any conflict, return False
                # Optimization: precomputed conflicts
                for j in core_conflicts[i]:
                    if mask & (1 << j):
                        return False
        return True
        
    # To speed up, we can generate only valid subsets via backtracking
    def enumerate_core(idx, current_mask, current_sum, current_indices):
        nonlocal best_global_sum, best_global_set
        
        if time.time() > deadline:
            return
            
        if idx == n_core:
            # Evaluate this core subset
            # Determine compatible rest indices
            # A rest index r is compatible if it does not conflict with ANY core index in current_mask
            
            # If current_mask is 0, all rest are available (subject to rest-rest conflicts)
            # If current_mask is not 0, filter rest
            
            # Build set of blocked rest indices
            blocked_rest = set()
            for c_idx in current_indices:
                blocked_rest.update(core_conflicting_rest[c_idx])
                
            # Available rest indices
            available_rest = [r for r in range(n_rest) if r not in blocked_rest]
            
            # Solve MWIS on available_rest using Greedy + LS
            rest_sum, rest_indices = solve_rest_greedy_ls(available_rest)
            
            total_sum = current_sum + rest_sum
            if total_sum > best_global_sum:
                best_global_sum = total_sum
                # Reconstruct set
                core_vals = [core_candidates[i] for i in current_indices]
                rest_vals = [rest_candidates[i] for i in rest_indices]
                best_global_set = core_vals + rest_vals
            return
        
        # Option 1: Exclude idx
        enumerate_core(idx + 1, current_mask, current_sum, current_indices)
        
        # Option 2: Include idx (if valid)
        # Check if idx conflicts with any in current_indices
        can_add = True
        for c_idx in current_indices:
            if idx in core_conflicts[c_idx]:
                can_add = False
                break
                
        if can_add:
            enumerate_core(idx + 1, current_mask | (1 << idx), current_sum + 1.0 / core_candidates[idx], current_indices + [idx])
            
    def solve_rest_greedy_ls(available_rest_indices):
        if not available_rest_indices:
            return 0.0, []
            
        # Sort by value ascending (to maximize 1/x)
        # available_rest_indices are indices into rest_candidates
        sorted_indices = sorted(available_rest_indices, key=lambda i: rest_candidates[i])
        
        selected = []
        selected_set = set()
        current_sum = 0.0
        
        # Greedy
        for i in sorted_indices:
            # Check compatibility with selected
            # rest_conflicts[i] contains indices that conflict with i
            if selected_set & rest_conflicts[i]:
                continue
            selected.append(i)
            selected_set.add(i)
            current_sum += 1.0 / rest_candidates[i]
            
        # Local Search: 1-Swap
        # Try to improve by swapping
        # We do a few passes
        # Since n_rest is small, we can afford a simple loop
        
        # Optimization: Only try swaps if time permits
        if time.time() < deadline - 5.0:
            improved = True
            while improved:
                improved = False
                # Try adding unselected
                # We already did greedy add.
                # Try swapping: remove i, add j
                # We want 1/j > 1/i => j < i
                # And j compatible with selected \ {i}
                
                # To keep it fast, we limit the scope of swaps
                # Just try a few random swaps or best-first?
                # Let's try a simple hill climb:
                # For each i in selected:
                #   For each j in available_rest_indices \ selected:
                #     if j < i and compatible:
                #       swap
                
                # This is O(N^2). N ~ 240. 57k ops.
                # Might be too slow if done for 32k cores.
                # Let's skip LS for speed, or do it only for the best few cores?
                # Or do it only if the core is small?
                
                # Let's try a very limited LS:
                # Just check if we can add any more? (Greedy did that)
                # Just check if we can swap one out for a better one?
                
                # Given the time constraint, we will skip complex LS on rest for every core.
                # The greedy is likely sufficient.
                # We can add a global LS on the final best set later?
                # No, the structure is different.
                
                break # Skip LS for now to ensure speed
                
        return current_sum, selected

    # Start enumeration
    enumerate_core(0, 0, 0.0, [])
    
    # If we have time, do a local search on the global best set?
    # The global best set is a list of values.
    # We can convert to indices, check validity, and try to improve.
    
    if time.time() < deadline - 5.0 and best_global_set:
        # Convert best_global_set to indices in all_candidates
        # This is a bit complex as we split core/rest.
        # Let's just return the best found.
        pass
        
    if not best_global_set:
        # Fallback: just pick 256? Or 2?
        # 2 conflicts with many.
        # 256 is safe with many.
        # Let's just return [256]
        return [256]
        
    return best_global_set
