"""Difference triangle set (7,5) solver with local search improvement."""
import random
import time

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct a difference triangle set: n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct
    across all rows. Minimize the largest entry (scope).
    
    Strategy:
    1. Greedy construction to get a valid starting set.
    2. Local search to reduce the scope.
    """
    deadline = time.time() + 115  # 150s wall clock, leave safety margin
    
    # Start with greedy construction
    rows = _greedy_construct(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    if time.time() > deadline:
        return best_rows
    
    # Local search to improve
    best_rows = _local_search(n_rows, k, best_rows, best_scope, deadline)
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Greedy construction: build rows one at a time, choosing smallest valid entries."""
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


def _get_all_diffs(rows):
    """Get all positive differences across all rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows):
    """Check if the set of rows is valid (all differences distinct across rows)."""
    diffs = []
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.append(d)
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _local_search(n_rows, k, rows, current_scope, deadline):
    """
    Local search to reduce scope. Try to move individual entries to smaller values
    while maintaining validity.
    """
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    rng = random.Random(42)
    
    # Precompute all differences for quick checking
    def get_diffs_for_row(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Build set of all used differences
    all_diffs = _get_all_diffs(best_rows)
    
    # Try to reduce each entry in each row
    # We'll do multiple passes
    max_iterations = 2000
    iteration = 0
    
    while iteration < max_iterations and time.time() < deadline:
        iteration += 1
        improved = False
        
        # Shuffle rows and positions for diversity
        row_indices = list(range(n_rows))
        rng.shuffle(row_indices)
        
        for ri in row_indices:
            if time.time() > deadline:
                break
            
            row = best_rows[ri]
            # Try to reduce each position (except position 0 which must be 0)
            for pi in range(len(row) - 1, 0, -1):  # Try from end to start
                if time.time() > deadline:
                    break
                
                old_val = row[pi]
                # Try values from old_val - 1 down to row[pi-1] + 1
                for new_val in range(old_val - 1, row[pi - 1], -1):
                    # Check if new_val works
                    # New differences involving position pi
                    old_diffs = get_diffs_for_row(row)
                    
                    # Remove old differences involving position pi
                    for i in range(pi):
                        old_diffs.discard(row[pi] - row[i])
                    for i in range(pi + 1, len(row)):
                        old_diffs.discard(row[i] - row[pi])
                    
                    # Add new differences
                    row[pi] = new_val
                    new_diffs = get_diffs_for_row(row)
                    
                    # Check for conflicts with other rows' differences
                    # We need to check against all differences in other rows
                    conflict = False
                    for other_ri in range(n_rows):
                        if other_ri == ri:
                            continue
                        other_diffs = get_diffs_for_row(best_rows[other_ri])
                        if new_diffs & other_diffs:
                            conflict = True
                            break
                    
                    if not conflict:
                        # Accept the change
                        # Update all_diffs
                        all_diffs = all_diffs - old_diffs | new_diffs
                        improved = True
                        break
                    else:
                        # Revert
                        row[pi] = old_val
                
                if improved:
                    break
            
            if improved:
                new_scope = _scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
    
    return best_rows


# EVOLVE-BLOCK-END
