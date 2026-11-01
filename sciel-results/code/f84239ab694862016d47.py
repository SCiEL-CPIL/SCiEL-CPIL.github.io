import time
import sys
import random
from itertools import combinations

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 110
    
    # The problem is for n_rows=7, k=5 (6 elements per row)
    # But the function signature is general.
    # We assume the evaluator passes n_rows=7, k=5.
    # We will implement a general solver but tuned for this case.
    
    # Number of differences per row
    num_diffs = (k + 1) * k // 2
    
    # 1. Generate a pool of candidate rows
    # We want rows with small maximum entries.
    # Best known scope is 112. Counting bound is 105.
    # We will generate rows up to a certain max, say 120.
    
    max_scope_limit = 120
    pool_size_target = 3000
    
    pool = []
    seen_rows = set()
    
    # Generate random rows
    # A row is a sorted list of k+1 integers starting with 0.
    # We need to ensure internal differences are distinct (Sidon set property).
    
    def generate_random_row(max_val):
        # Try to generate a valid row with max <= max_val
        for _ in range(100):
            # Randomly pick k elements from 1..max_val
            # Ensure they are distinct and > 0
            candidates = random.sample(range(1, max_val + 1), k)
            row = [0] + sorted(candidates)
            
            # Check internal differences
            diffs = set()
            valid = True
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            
            if valid:
                # Check if we've seen this row before
                row_tuple = tuple(row)
                if row_tuple not in seen_rows:
                    seen_rows.add(row_tuple)
                    return row, diffs, max(row)
        return None

    # Generate pool
    # We bias towards smaller max values to help find minimal scope solutions
    # We can generate rows in batches with increasing max_val
    current_max = 105
    while len(pool) < pool_size_target and current_max <= max_scope_limit:
        for _ in range(100):
            if len(pool) >= pool_size_target:
                break
            if time.time() > deadline - 30:
                break
            res = generate_random_row(current_max)
            if res:
                pool.append(res)
        current_max += 5
        
    # If pool is still small, generate with larger max
    if len(pool) < pool_size_target:
        while len(pool) < pool_size_target and time.time() < deadline - 30:
            res = generate_random_row(max_scope_limit)
            if res:
                pool.append(res)

    if not pool:
        return []

    # 2. Search for an independent set of size n_rows
    # We want to minimize the maximum entry.
    # We can iterate over possible scope M from 105 to max_scope_limit
    # and check if a valid set exists with all rows having max <= M.
    
    # Sort pool by max entry to make filtering easier
    pool.sort(key=lambda x: x[2])
    
    best_solution = None
    best_scope = float('inf')
    
    # To speed up search, we can use backtracking
    # State: selected rows, used diffs, index in pool
    # We want to find n_rows rows.
    
    # Optimization: Group pool by max entry? Or just filter.
    # Since n_rows is small (7), backtracking is feasible.
    
    def find_independent_set(pool_subset, n_needed, used_diffs, selected_rows, current_best_scope):
        if len(selected_rows) == n_needed:
            scope = max(max(r[0]) for r in selected_rows)
            if scope < current_best_scope[0]:
                current_best_scope[0] = scope
                current_best_scope[1] = [r[0] for r in selected_rows]
            return True
            
        if not pool_subset:
            return False
            
        # Pruning: If the smallest possible scope we can achieve exceeds current best, stop
        # The smallest possible max entry for the remaining rows is the max of the first available row's max?
        # Not exactly, but we can use a lower bound.
        
        # Heuristic: Try rows with smaller max entries first
        # The pool is already sorted by max entry.
        
        # We need to pick n_needed rows from pool_subset
        # such that their diffs don't conflict with used_diffs and each other.
        
        # Simple backtracking
        # To reduce branching, we can skip rows that conflict with used_diffs
        # We can also use a "forward checking" style: 
        # If we pick a row, we update used_diffs.
        
        # Let's iterate through candidates
        # To avoid trying the same set of rows in different orders, 
        # we enforce an order: index in pool_subset must increase.
        
        def backtrack(start_idx, n_needed, used_diffs, selected_rows):
            if len(selected_rows) == n_needed:
                scope = max(max(r[0]) for r in selected_rows)
                if scope < current_best_scope[0]:
                    current_best_scope[0] = scope
                    current_best_scope[1] = [r[0] for r in selected_rows]
                return True
                
            if start_idx >= len(pool_subset):
                return False
                
            # Lower bound pruning
            # If we have picked p rows, we need n_needed - p more.
            # The max entry of the solution will be at least the max entry of the largest picked row so far.
            # Also, the next row we pick will have max entry >= pool_subset[start_idx][2].
            
            # Try each candidate from start_idx onwards
            # Optimization: If pool_subset[i][2] >= current_best_scope[0], 
            # and we haven't filled all rows, can we skip?
            # No, because a later row might have a smaller max? No, pool is sorted.
            # So if pool_subset[i][2] >= current_best_scope[0], 
            # any solution including this row or later will have scope >= current_best_scope[0].
            # So we can break if we are looking for strictly better?
            # But we want to find *any* solution first, then improve.
            # Actually, since we iterate M from 105 upwards, we just need to find *a* solution for the current M.
            
            # Let's restructure: 
            # Instead of one big search, we iterate M from 105 to 120.
            # For each M, we filter pool to rows with max <= M.
            # Then we try to find *any* independent set of size n_rows.
            # If found, we return it.
            # This guarantees minimal scope if we iterate M in increasing order.
            
            return False
            
        # The previous nested function structure is getting complex.
        # Let's implement the outer loop for M.
        
    # Iterate M from 105 to max_scope_limit
    # We can optimize by only checking M values that appear in the pool
    possible_scopes = sorted(set(r[2] for r in pool))
    
    for M in possible_scopes:
        if M > max_scope_limit:
            break
        if time.time() > deadline - 10:
            break
            
        # Filter pool for this M
        # Since pool is sorted, we can find the cutoff index
        # But we need to pass the filtered list to the search.
        # Creating a new list is O(P). P is ~3000. This is fine.
        
        filtered_pool = [r for r in pool if r[2] <= M]
        
        if len(filtered_pool) < n_rows:
            continue
            
        # Try to find an independent set of size n_rows in filtered_pool
        # Use backtracking
        found = False
        best_set = None
        
        # To make backtracking faster, we can shuffle the filtered pool?
        # No, we want to find *a* solution. The order doesn't matter for existence.
        # But for speed, we might want to try "good" rows first.
        # Rows with smaller max are better. They are already first.
        
        # We can also pre-compute conflicts?
        # For 3000 rows, a 3000x3000 bitset is 1.1MB. Feasible.
        # But checking on the fly with sets is also fast (15 elements).
        
        def search(iset, n_needed, used_diffs, selected, start_idx):
            if len(selected) == n_needed:
                return True, selected[:]
            if start_idx >= len(iset):
                return False, None
                
            # Pruning: If remaining items < n_needed - len(selected), fail
            if len(iset) - start_idx < n_needed - len(selected):
                return False, None
                
            for i in range(start_idx, len(iset)):
                row, diffs, max_val = iset[i]
                # Check conflict with used_diffs
                if used_diffs & diffs:
                    continue
                    
                # Check conflict with other selected rows?
                # used_diffs contains all diffs from selected rows.
                # So if used_diffs & diffs is empty, it's compatible with all selected.
                
                # Recurse
                new_used = used_diffs | diffs
                new_selected = selected + [(row, diffs, max_val)]
                
                # Optimization: Forward checking?
                # If we pick this row, do we still have enough candidates left?
                # The remaining candidates are iset[i+1:].
                # We need n_needed - len(new_selected) more.
                # If len(iset) - (i+1) < n_needed - len(new_selected), fail.
                if len(iset) - (i + 1) < n_needed - len(new_selected):
                    continue
                    
                found_sub, res_sub = search(iset, n_needed, new_used, new_selected, i + 1)
                if found_sub:
                    return True, res_sub
                    
            return False, None
            
        success, solution = search(filtered_pool, n_rows, set(), [], 0)
        
        if success:
            # Extract rows
            best_solution = [s[0] for s in solution]
            best_scope = M
            break # Found minimal scope
            
    if not best_solution:
        # Fallback: Return any valid set found or empty
        # If we didn't find one with the strict M loop, maybe the pool was insufficient
        # We can try a less strict search
        # But for now, return best effort or empty
        return []
        
    return best_solution

