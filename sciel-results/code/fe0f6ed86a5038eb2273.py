"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set: n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct
    across all rows. Minimize the largest entry (scope).
    
    Uses a combination of:
    1. Backtracking search with forward-looking heuristics for initial construction
    2. Simulated annealing to refine and improve the scope
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Total number of differences needed: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Simple counting bound: we need at least total_diffs distinct positive differences
    # The largest difference in any row is at most m (the scope)
    # Number of available differences in [1, m] is m
    # So m >= total_diffs, but also each row's max element contributes differences
    # A better bound: if scope is m, we can have differences 1..m, so m >= total_diffs
    
    # Start with greedy construction to get a valid solution
    best_rows = _greedy_construct(n_rows, k)
    best_scope = _get_scope(best_rows)
    
    if time.time() > deadline:
        return best_rows
    
    # Try to improve using backtracking with heuristic
    # This attempts to build rows with a tighter scope
    improved = _backtracking_search(n_rows, k, best_scope, deadline)
    if improved is not None:
        scope = _get_scope(improved)
        if scope < best_scope:
            best_rows = improved
            best_scope = scope
    
    if time.time() > deadline:
        return best_rows
    
    # Simulated annealing to further reduce scope
    best_rows, best_scope = _simulated_annealing(
        best_rows, n_rows, k, best_scope, deadline
    )
    
    return best_rows


def _get_scope(rows):
    """Get the maximum entry across all rows."""
    return max(max(row) for row in rows)


def _is_valid(rows, n_rows, k):
    """Check if the rows form a valid difference triangle set."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff = row[j] - row[i]
                if diff <= 0:
                    return False
                if diff in used:
                    return False
                used.add(diff)
    return len(rows) == n_rows


def _greedy_construct(n_rows, k):
    """Greedy construction: build each row with smallest possible marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _backtracking_search(n_rows, k, max_scope, deadline):
    """
    Try to construct rows with scope <= max_scope - 1 using backtracking.
    Uses forward-looking heuristic to choose next element.
    """
    target_scope = max_scope - 1
    if target_scope < k:  # minimum possible scope for k+1 elements starting at 0
        return None
    
    used = set()
    rows = []
    
    def backtrack(row_idx, row):
        nonlocal used, rows, target_scope
        
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if len(row) == k + 1:
            rows.append(row)
            return backtrack(row_idx + 1, [0])
        
        # Determine candidate range
        last_val = row[-1]
        start_candidate = last_val + 1
        
        # Try candidates in order, but use a heuristic to prioritize
        # candidates that leave more room for future differences
        candidates = list(range(start_candidate, target_scope + 1))
        
        # Heuristic: prefer candidates that minimize the number of "wasted"
        # small differences that could be used later
        # For now, try in natural order but limit search depth
        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            
            # Forward check: can we still complete this row?
            # Remaining elements needed: k + 1 - len(row) - 1
            remaining = k + 1 - len(row) - 1
            if remaining > 0:
                # We need at least 'remaining' more distinct differences
                # from the new position x to the future positions
                # The minimum next position is x+1, so minimum new diff is 1
                # But we also need these diffs to not conflict with used
                # Simple check: there must be at least 'remaining' unused diffs
                # in the range [1, target_scope - x] (approximate)
                available = 0
                for d in range(1, target_scope - x + 1):
                    if (x + d) <= target_scope and (x - (x - d)) not in used:
                        # This is getting complex; use a simpler heuristic
                        pass
                # Skip complex forward check for now, rely on backtracking
                pass
            
            used |= new_diffs
            row.append(x)
            
            if backtrack(row_idx, row):
                return True
            
            row.pop()
            used -= new_diffs
        
        return False
    
    # Try to build with target_scope
    # First row starts with [0]
    result = backtrack(0, [0])
    
    if result:
        # Verify
        if _is_valid(rows, n_rows, k):
            return rows
    return None


def _simulated_annealing(rows, n_rows, k, initial_scope, deadline):
    """
    Simulated annealing to reduce the scope of a valid difference triangle set.
    
    State: list of rows
    Move: modify one element of one row (keeping it increasing and starting at 0)
    Cost: scope (max element), plus a penalty for invalidity
    """
    import math
    
    best_rows = [row[:] for row in rows]
    best_scope = initial_scope
    current_rows = [row[:] for row in rows]
    
    # Initial temperature
    T = max(10.0, initial_scope * 0.1)
    cooling_rate = 0.995
    
    rng = random.Random(42)
    
    def get_cost(rows):
        """Cost = scope + large penalty if invalid."""
        scope = max(max(row) for row in rows)
        if not _is_valid(rows, n_rows, k):
            return scope + 10000
        return scope
    
    def get_scope_only(rows):
        return max(max(row) for row in rows)
    
    current_cost = get_cost(current_rows)
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick a random row and position to modify
        r_idx = rng.randint(0, n_rows - 1)
        
        if r_idx == 0:
            # Row 0 is fixed at position 0
            pos_options = list(range(1, k + 1))
        else:
            pos_options = list(range(1, k + 1))
        
        pos = rng.choice(pos_options)
        old_val = current_rows[r_idx][pos]
        
        # Propose a new value
        # Bias towards smaller values to reduce scope
        if rng.random() < 0.7:
            # Try a smaller value
            min_allowed = current_rows[r_idx][pos - 1] + 1
            max_allowed = old_val - 1
            if max_allowed >= min_allowed:
                new_val = rng.randint(min_allowed, max_allowed)
            else:
                continue
        else:
            # Try a larger value (to escape local minima)
            min_allowed = old_val + 1
            max_allowed = initial_scope + 20
            if max_allowed >= min_allowed:
                new_val = rng.randint(min_allowed, max_allowed)
            else:
                continue
        
        # Check if the new value maintains increasing order
        if pos > 0 and new_val <= current_rows[r_idx][pos - 1]:
            continue
        if pos < k and new_val >= current_rows[r_idx][pos + 1]:
            continue
        
        # Apply the move
        current_rows[r_idx][pos] = new_val
        new_cost = get_cost(current_rows)
        
        delta_cost = new_cost - current_cost
        
        # Metropolis acceptance
        if delta_cost <= 0 or (T > 0 and rng.random() < math.exp(-delta_cost / T)):
            current_cost = new_cost
            current_scope = get_scope_only(current_rows)
            
            if _is_valid(current_rows, n_rows, k) and current_scope < best_scope:
                best_rows = [row[:] for row in current_rows]
                best_scope = current_scope
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_rows[r_idx][pos] = old_val
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        
        iterations += 1
        
        # Early termination if stuck for too long
        if no_improve_count > max_no_improve and T < 1.0:
            break
        
        # Check time periodically
        if iterations % 1000 == 0 and time.time() > deadline:
            break
    
    return best_rows, best_scope


# EVOLVE-BLOCK-END
