"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import sys
import time

def construct(n_rows, k):
    # Difference triangle set: choose n_rows rows of k+1 increasing integers starting at 0
    # such that all positive differences a_j - a_i (i < j) are distinct across all rows.
    # Minimize the largest entry.
    
    best_rows = None
    best_max = float('inf')
    
    start_time = time.time()
    max_time = 110.0  # Leave some time for overhead
    
    # Approach: Randomized Greedy Construction followed by Local Search
    # This is distinct from the backtracking approach.
    
    for attempt in range(5000):
        if time.time() - start_time > max_time:
            break
            
        # 1. Generate a random valid solution using randomized greedy
        rows = _random_greedy(n_rows, k)
        
        if rows is None:
            continue
            
        # 2. Improve it using local search
        rows, max_val = _local_search(rows, n_rows, k)
        
        if max_val < best_max:
            best_max = max_val
            best_rows = rows
            
            # If we hit the theoretical lower bound, we can stop
            # Counting bound: n_rows * (k+1)*k/2 distinct diffs.
            # For n_rows=4, k=7: 112.
            if best_max <= n_rows * (k + 1) * k // 2:
                break
                
    if best_rows is None:
        # Fallback to a simple greedy if no solution found
        best_rows = _greedy_fallback(n_rows, k)
        
    return best_rows

def _random_greedy(n_rows, k):
    """
    Generates a valid configuration using a randomized greedy strategy.
    For each new element, picks a random valid candidate from a range.
    """
    rows = []
    used_diffs = set()
    
    for r in range(n_rows):
        row = [0]
        # To ensure we don't go too high, we can limit the search range for candidates
        # but we need to be careful not to make it impossible to complete the row.
        
        for pos in range(1, k + 1):
            # Determine a reasonable upper bound for the next element
            # A heuristic: the max value in the current row + some slack
            # Or just a fixed window relative to the current max.
            current_max = row[-1]
            
            # We need to find a valid x > current_max
            # Let's try candidates in a window [current_max + 1, current_max + 50]
            # If we fail, expand the window.
            
            found = False
            search_limit = 50
            while not found and search_limit < 500:
                # Generate candidates in the window
                # To randomize, we can shuffle a list of candidates or pick randomly
                # Picking randomly from a range is efficient.
                
                # We need x such that all diffs x-a for a in row are not in used_diffs
                # and x > current_max
                
                # Let's try 20 random candidates in the window
                for _ in range(20):
                    x = random.randint(current_max + 1, current_max + search_limit)
                    
                    # Check validity
                    valid = True
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    search_limit *= 2
                    
            if not found:
                return None # Failed to find a valid element
            
        rows.append(row)
        
    return rows

def _local_search(rows, n_rows, k):
    """
    Attempts to minimize the maximum element by locally reducing elements.
    Repeatedly scans for any element that can be reduced while maintaining validity.
    """
    # Copy rows to avoid modifying the input if we want to be safe, 
    # but we return a new list anyway.
    current_rows = [list(r) for r in rows]
    used_diffs = set()
    
    # Rebuild used_diffs
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
                
    improved = True
    max_iterations = 1000
    
    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # Try to reduce each element in each row (except the first 0)
        # We process in reverse order of magnitude? Or just row by row?
        # Let's try to reduce the largest elements first to get quick wins on the max.
        
        # Create a list of (value, row_idx, col_idx) and sort by value descending
        candidates = []
        for r_idx in range(n_rows):
            for c_idx in range(1, len(current_rows[r_idx])):
                candidates.append((current_rows[r_idx][c_idx], r_idx, c_idx))
                
        candidates.sort(reverse=True)
        
        for val, r_idx, c_idx in candidates:
            if val != current_rows[r_idx][c_idx]:
                continue # Element has been changed in a previous step of this iteration
                
            row = current_rows[r_idx]
            # Try to find a smaller valid value for row[c_idx]
            # The new value must be > row[c_idx-1] and < val
            min_val = row[c_idx - 1] + 1
            max_val = val - 1
            
            if min_val > max_val:
                continue
                
            # Try to find the smallest valid value in [min_val, max_val]
            # We can iterate from min_val upwards to find the absolute smallest,
            # which is best for minimization.
            
            # Optimization: We only need to check values that don't conflict.
            # Since the range might be small, linear scan is fine.
            
            found_new_val = -1
            for v in range(min_val, max_val + 1):
                # Check if v is valid
                # We need to remove old diffs involving row[c_idx] and add new ones
                # But we can just check against the global used_diffs excluding those from this row?
                # No, easier: 
                # 1. Remove all diffs involving row[c_idx] from a temporary set?
                # 2. Or just check if new diffs conflict with (used_diffs - diffs_involving_row_c_idx)
                
                # Let's compute the set of diffs that involve row[c_idx] in the current row
                old_diffs_to_remove = set()
                for i in range(len(row)):
                    if i != c_idx:
                        d = abs(row[c_idx] - row[i])
                        old_diffs_to_remove.add(d)
                        
                # The "other" diffs are used_diffs - old_diffs_to_remove
                # We need to check if any new diff conflicts with "other" diffs
                # AND if new diffs are unique among themselves (which they are if row is strictly increasing? 
                # No, diffs within the same row must be distinct too. 
                # Wait, the problem says "all positive differences ... are distinct over all rows".
                # This implies diffs within a row must also be distinct?
                # "distinct over all rows" usually means the set of all diffs from all rows has no duplicates.
                # So yes, diffs within a row must be distinct as well.
                
                # So we need to check:
                # 1. New diffs are distinct from each other.
                # 2. New diffs are not in (used_diffs - old_diffs_to_remove).
                
                new_diffs = set()
                valid = True
                for i in range(len(row)):
                    if i != c_idx:
                        d = v - row[i]
                        if d <= 0:
                             # This shouldn't happen if v > row[c_idx-1] and we assume increasing order?
                             # If i > c_idx, v < row[i] is required for increasing order.
                             # So we must ensure v < row[i] for all i > c_idx.
                             # And v > row[i] for all i < c_idx.
                             pass
                        if d in used_diffs and d not in old_diffs_to_remove:
                            valid = False
                            break
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                        
                # Check ordering constraints
                if valid:
                    if c_idx > 0 and v <= row[c_idx - 1]:
                        valid = False
                    if c_idx < len(row) - 1 and v >= row[c_idx + 1]:
                        valid = False
                        
                if valid:
                    found_new_val = v
                    break # We want the smallest valid v to minimize the max as much as possible locally
                    
            if found_new_val != -1:
                # Update the row and used_diffs
                old_val = row[c_idx]
                row[c_idx] = found_new_val
                
                # Update used_diffs
                # Remove old diffs
                for d in old_diffs_to_remove:
                    used_diffs.discard(d)
                # Add new diffs
                used_diffs.update(new_diffs)
                
                improved = True
                # We break out of the candidate loop because the set of candidates is stale?
                # Or we can continue, but we need to be careful.
                # Let's restart the scan to be safe and simple.
                break
                
    max_val = max(max(row) for row in current_rows)
    return current_rows, max_val

def _greedy_fallback(n_rows, k):
    """Simple greedy to ensure we return *something* valid."""
    rows = []
    used_diffs = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
