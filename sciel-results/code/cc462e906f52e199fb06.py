import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Compute set of positive differences in a row."""
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

def _validate_solution(rows, n_rows, k):
    """Check if all rows are valid and all cross-row differences are distinct."""
    if len(rows) != n_rows:
        return False
    all_used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        rd = _row_diffs(row)
        if rd & all_used:
            return False
        all_used |= rd
    return True

def _greedy_init(n_rows, k):
    """
    Greedy: build each row with the smallest possible next element that avoids all used diffs.
    """
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_d = {x - a for a in row}
                if not (new_d & used) and len(new_d) == len(row):
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows

def _generate_rows(k, max_scope):
    """
    Generate all valid Sidon rows of length k+1 starting at 0 with max entry <= max_scope.
    Returns a list of tuples (row_tuple, mask_int).
    """
    rows = []
    
    def backtrack(current_row, used_mask, depth):
        if depth == k + 1:
            # Convert to tuple for hashability and storage
            rows.append((tuple(current_row), used_mask))
            return
        
        last = current_row[-1]
        # Next value must be > last
        # Max value is max_scope
        # Also, we need enough room for remaining elements?
        # remaining = k + 1 - depth
        # last + remaining - 1 <= max_scope
        
        start_val = last + 1
        end_val = max_scope
        
        for v in range(start_val, end_val + 1):
            # Calculate new diffs
            new_diffs_mask = 0
            valid = True
            for a in current_row:
                d = v - a
                bit = 1 << d
                if used_mask & bit:
                    valid = False
                    break
                new_diffs_mask |= bit
            
            if not valid:
                continue
            
            # Check internal collisions in the new row?
            # The mask check `used_mask & bit` ensures no diff is repeated in the row 
            # IF `used_mask` contained the diffs of the current row.
            # Yes, `used_mask` is passed as the mask of diffs of `current_row`.
            # So if `v-a` is already in `used_mask`, it means collision.
            
            backtrack(current_row + [v], used_mask | new_diffs_mask, depth + 1)

    backtrack([0], 0, 1)
    return rows

def _find_min_scope(n_rows, k, initial_upper_bound, time_limit):
    """
    Find the set of n_rows rows with minimal max entry.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # 1. Generate all rows up to initial_upper_bound
    # Note: If the optimal solution is smaller, we might generate extra rows, 
    # but it's necessary to have the pool.
    all_rows = _generate_rows(k, initial_upper_bound)
    
    if not all_rows:
        return None, initial_upper_bound

    # Sort by max entry (which is the last element of the tuple)
    # all_rows[i] is (row_tuple, mask)
    # row_tuple[-1] is the max
    all_rows.sort(key=lambda x: x[0][-1])
    
    best_rows = None
    best_max = initial_upper_bound
    
    # DFS state
    selected_rows = []
    used_mask = 0
    
    def dfs(idx, current_max, current_used_mask, selected):
        nonlocal best_rows, best_max
        
        if time.time() > deadline:
            return
            
        if len(selected) == n_rows:
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in selected]
            return
        
        # Pruning: 
        # 1. If we can't pick enough rows from the remaining candidates, return.
        # Remaining candidates are from idx to end.
        # But we must filter by conflict.
        # Simple bound: if len(all_rows) - idx < n_rows - len(selected), return.
        if len(all_rows) - idx < n_rows - len(selected):
            return
            
        # Heuristic: We are iterating in increasing max order.
        # If current_max is already >= best_max, we can prune?
        # Not necessarily, because we might pick rows with smaller max later?
        # No, all_rows is sorted by max. So all_rows[idx][0][-1] >= current_max is not guaranteed 
        # if we skipped some? No, we are picking from idx onwards.
        # Actually, if we pick a row with max M, current_max becomes M.
        # Since we iterate in increasing order, the max of any future row will be >= M.
        # So if current_max >= best_max, we can prune the whole branch.
        if current_max >= best_max:
            return
        
        # Iterate candidates from idx
        for i in range(idx, len(all_rows)):
            row_tuple, mask = all_rows[i]
            
            # Check conflict
            if mask & current_used_mask:
                continue
                
            new_max = max(current_max, row_tuple[-1])
            
            # Prune if new_max >= best_max
            if new_max >= best_max:
                # Since all_rows is sorted, all subsequent rows have max >= row_tuple[-1] = new_max (if row_tuple[-1] > current_max)
                # Or if current_max > row_tuple[-1], new_max = current_max.
                # If new_max >= best_max, we can break?
                # If new_max == current_max and current_max >= best_max, we already pruned above.
                # If new_max > current_max, then subsequent rows have max >= new_max, so they will also have new_max >= best_max.
                # So we can break.
                break
                
            # Recurse
            dfs(i + 1, new_max, current_used_mask | mask, selected + [row_tuple])
            
            # Check time
            if time.time() > deadline:
                return

    dfs(0, 0, 0, [])
    
    if best_rows:
        return best_rows, best_max
    else:
        # Fallback to greedy if DFS fails or times out without finding better
        greedy_rows = _greedy_init(n_rows, k)
        return greedy_rows, max(max(r) for r in greedy_rows)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initial Greedy
    rows = _greedy_init(n_rows, k)
    initial_max = max(max(r) for r in rows)
    
    # 2. Try to improve with MIS/Branch-and-Bound
    # We know the bound is around 105-112. 
    # Greedy might give ~115-120.
    # We generate rows up to initial_max.
    # If initial_max is very high, generation might be slow.
    # Let's cap the generation scope to min(initial_max, 120) to be safe.
    # If the optimal is > 120, we return greedy. But 112 is best known.
    gen_scope = min(initial_max, 125) 
    
    try:
        improved_rows, improved_max = _find_min_scope(n_rows, k, gen_scope, deadline - start_time - 5)
        if improved_rows and improved_max < initial_max:
            rows = improved_rows
    except Exception:
        pass # Keep greedy result

    return rows

# EVOLVE-BLOCK-END
