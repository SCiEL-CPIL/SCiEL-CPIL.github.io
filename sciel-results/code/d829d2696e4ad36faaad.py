"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill

    # Total number of differences needed: n_rows * k(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2

    # Best solution found so far
    best_rows = None
    best_max = float('inf')

    # First, get a baseline using greedy
    baseline_rows = _pure_greedy(n_rows, k)
    if baseline_rows is not None:
        best_rows = baseline_rows
        best_max = max(max(row) for row in baseline_rows)

    # Try randomized backtracking search with multiple restarts
    # We'll try to find a solution with max entry < best_max
    random.seed(42)

    # Upper bound estimate: start from best_max and try to improve
    # The counting bound is 112 for (4,7), but best known is 134
    # Let's try various upper bounds
    
    # Try to find solutions with decreasing upper bounds
    # Start from best_max - 1 and go down
    # But first, let's try a few randomized searches with the current best_max as upper bound
    
    attempts = 0
    max_attempts = 2000  # Will be limited by time
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Try with a slightly tighter bound
        # Start with best_max - 1, then try lower
        target_max = best_max - 1 - (attempt % 20)  # Gradually tighten
        if target_max < 0:
            target_max = 0
        
        rows = _backtracking_search(n_rows, k, target_max, time.time(), deadline)
        
        if rows is not None:
            row_max = max(max(row) for row in rows)
            if row_max < best_max:
                best_max = row_max
                best_rows = rows
                # Reset the target to try even tighter
                attempts = 0
        else:
            attempts += 1
    
    # Final greedy pass with the best_max as upper bound to see if we can do better
    if best_rows is None:
        # Fallback: return greedy result
        best_rows = _pure_greedy(n_rows, k)
    
    return best_rows


def _pure_greedy(n_rows, k):
    """Greedy construction: each row starts at 0 and takes the smallest next mark."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _backtracking_search(n_rows, k, max_val, start_time, deadline):
    """
    Backtracking search to find n_rows rows with max entry <= max_val.
    Uses randomized order for exploring candidates.
    Returns list of rows if found, else None.
    """
    used = set()
    rows = []
    
    def _build_row(row_idx, current_row, current_x):
        """Build one row using backtracking."""
        if len(current_row) == k + 1:
            return True
        
        # Time check
        if time.time() > deadline:
            return False
        
        # Try candidates from current_x+1 to max_val
        # Randomize order for diversity
        candidates = list(range(current_x + 1, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in current_row}
            if new_diffs & used:
                continue
            
            # Try adding x
            current_row.append(x)
            used.update(new_diffs)
            
            if _build_row(row_idx, current_row, x):
                return True
            
            # Backtrack
            current_row.pop()
            used.difference_update(new_diffs)
        
        return False
    
    for i in range(n_rows):
        row = [0]
        if not _build_row(i, row, 0):
            return None
        rows.append(row)
    
    return rows


def _backtracking_search_v2(n_rows, k, max_val, start_time, deadline):
    """
    Improved backtracking with better heuristics.
    """
    used = set()
    rows = []
    
    # For each row, we need to choose k more values after 0
    # Total positions to fill: n_rows * k
    
    def _solve(row_idx, pos_in_row, current_row, last_val):
        if time.time() > deadline:
            return False
        
        if pos_in_row == k + 1:
            rows.append(current_row)
            return _solve(row_idx + 1, 0, [0], 0)
        
        if row_idx == n_rows:
            return True
        
        # Determine range for next value
        start = last_val + 1
        end = max_val
        
        # Heuristic: try values that create "popular" differences first
        # or spread out values
        candidates = list(range(start, end + 1))
        
        # Shuffle for randomness
        random.shuffle(candidates)
        
        # Try each candidate
        for x in candidates:
            new_diffs = {x - a for a in current_row}
            if new_diffs & used:
                continue
            
            current_row.append(x)
            used.update(new_diffs)
            
            if _solve(row_idx, pos_in_row + 1, current_row, x):
                return True
            
            current_row.pop()
            used.difference_update(new_diffs)
        
        return False
    
    # We need to manage the rows list properly
    # Let's restructure: build all rows sequentially
    rows = []
    used = set()
    
    def _build_all_rows(row_idx, current_row, pos, last_val):
        if time.time() > deadline:
            return False
        
        if pos == k + 1:
            # Row complete
            rows.append(list(current_row))
            if row_idx + 1 == n_rows:
                return True
            return _build_all_rows(row_idx + 1, [0], 0, 0)
        
        start = last_val + 1
        end = max_val
        
        candidates = list(range(start, end + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in current_row}
            if new_diffs & used:
                continue
            
            current_row.append(x)
            used.update(new_diffs)
            
            if _build_all_rows(row_idx, current_row, pos + 1, x):
                return True
            
            current_row.pop()
            used.difference_update(new_diffs)
        
        return False
    
    if _build_all_rows(0, [0], 0, 0):
        return rows
    
    return None
# EVOLVE-BLOCK-END
