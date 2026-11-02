"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import sys
import time
import random

# EVOLVE-BLOCK-START
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    
    Uses a randomized backtracking search with multiple restarts for each scope m.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Lower bound: m >= total_diffs
    start_m = total_diffs
    
    # We will try m values starting from start_m
    # For each m, we try several random restarts
    max_m_tries = 50
    restarts_per_m = 10
    node_limit = 50000  # Limit search depth/width to prevent timeouts
    
    for i in range(max_m_tries):
        current_m = start_m + i
        
        # Try multiple random restarts for this m
        found = False
        for _ in range(restarts_per_m):
            result = _backtracking_construct_random(n_rows, k, current_m, node_limit)
            if result is not None:
                return result
        
    # Fallback: use the greedy approach if backtracking fails
    return _greedy_construct(n_rows, k)


def _backtracking_construct_random(n_rows, k, max_val, node_limit):
    """
    Randomized backtracking search to construct n_rows rows of k+1 integers in [0, max_val]
    starting at 0, with all differences distinct across all rows.
    """
    used_diffs = set()
    rows = []
    nodes_visited = [0]  # Use list to allow mutation in closure
    
    def build_row(row_idx, current_row, remaining):
        """
        Build a single row starting at 0.
        current_row: list of placed values so far (starts with [0])
        remaining: number of more values to place
        """
        if nodes_visited[0] > node_limit:
            return False
        
        if remaining == 0:
            return True
        
        last_val = current_row[-1]
        
        # Generate candidates
        # We need to pick 'remaining' values from (last_val, max_val]
        # such that they are strictly increasing.
        # To allow randomization, we can shuffle the candidate pool or pick randomly.
        # However, simple shuffling of the range is expensive.
        # Instead, we can randomly permute the order in which we try candidates,
        # or use a heuristic: try candidates that are "far" from existing ones?
        # For now, let's just shuffle the range [last_val+1, max_val]
        
        candidates = list(range(last_val + 1, max_val + 1))
        random.shuffle(candidates)
        
        for candidate in candidates:
            nodes_visited[0] += 1
            if nodes_visited[0] > node_limit:
                return False
            
            # Check if all differences are unused
            valid = True
            new_diffs = []
            for a in current_row:
                d = candidate - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Pruning: check if we can still complete the row
            # We need 'remaining - 1' more values after this one.
            # The minimum span required for 'r' points is roughly r*(r-1)/2? No, just r.
            # But they must be distinct integers > candidate.
            # So we need at least 'remaining - 1' integers in (candidate, max_val].
            if max_val - candidate < remaining - 1:
                continue  # Not enough room, try next candidate (which is larger, so this will fail for all larger too? No, larger candidate leaves LESS room. So if this fails, all larger candidates fail.)
                # Wait, candidates are shuffled. So we can't break. We must continue.
                # Actually, if candidate is too large, it fails. If candidate is small, it passes.
                # Since candidates are shuffled, we can't break on this condition easily.
            
            # Place the candidate
            current_row.append(candidate)
            used_diffs.update(new_diffs)
            
            if build_row(row_idx, current_row, remaining - 1):
                return True
            
            # Backtrack
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    # Build each row
    # To add more randomization, we can shuffle the order in which we build rows?
    # The rows are indistinguishable in the final set, so order doesn't matter for validity.
    # But the search space might be different. Let's stick to sequential for now, 
    # but the randomization in candidates helps.
    
    for row_idx in range(n_rows):
        current_row = [0]
        if not build_row(row_idx, current_row, k):
            return None
        rows.append(current_row)
    
    return rows


def _greedy_construct(n_rows, k):
    """Greedy fallback: same as the original seed program."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
