"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _validate(rows, k):
    """Check that rows are valid: each row has k+1 increasing ints starting at 0,
    and all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _score(rows):
    """Return the largest entry (scope)."""
    return max(max(row) for row in rows)

def _greedy_construct(n_rows, k, deadline=None):
    """Greedy rulers: each row starts at 0 and takes the smallest next mark whose differences
    to the earlier marks of the row are all still unused by any row."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _backtrack_find(n_rows, k, deadline=None):
    """Backtracking search to find rows with minimal max entry.
    Tries to find a valid configuration where all rows have max entry <= bound.
    Uses a deadline to stop early."""
    # We'll search for a specific bound M and try to fit n_rows rows of k+1 elements
    # starting at 0, strictly increasing, all differences distinct across all rows.
    
    # Start with a reasonable upper bound from greedy
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_max = max(max(row) for row in greedy_rows)
    
    # Try to find a better solution by backtracking
    # We'll try bounds from greedy_max-1 downward
    best_rows = greedy_rows
    best_max = greedy_max
    
    for M in range(greedy_max - 1, 0, -1):
        if deadline and time.time() > deadline:
            break
        # Try to find n_rows rows with max entry <= M
        rows = _find_rows(n_rows, k, M, deadline)
        if rows is not None:
            best_rows = rows
            best_max = M
            # If we found it, we can stop (we want minimal M)
            break
    
    return best_rows

def _find_rows(n_rows, k, M, deadline=None):
    """Try to find n_rows rows of k+1 elements (starting at 0, increasing, max <= M)
    with all differences distinct across all rows. Returns None if not found."""
    used = set()
    rows = []
    
    def backtrack(row_idx, row, x):
        if deadline and time.time() > deadline:
            return False
        if row_idx == n_rows:
            return True
        if len(row) == k + 1:
            rows.append(row[:])
            return backtrack(row_idx + 1, [0], 1)
        
        # Try next value for this row
        for nx in range(x, M + 1):
            if deadline and time.time() > deadline:
                return False
            new_diffs = {nx - a for a in row}
            if not new_diffs & used:
                row.append(nx)
                used.update(new_diffs)
                if backtrack(row_idx, row, nx + 1):
                    return True
                row.pop()
                used.difference_update(new_diffs)
        return False
    
    if backtrack(0, [0], 1):
        return rows
    return None

def _local_search(rows, k, deadline=None):
    """Local search: try to reduce the max entry by swapping or shifting elements."""
    current_rows = [row[:] for row in rows]
    current_max = max(max(row) for row in current_rows)
    best_rows = current_rows
    best_max = current_max
    
    # Compute all used differences
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    # Try to improve by reducing the largest element
    # This is a simple hill-climbing approach
    for iteration in range(1000):
        if deadline and time.time() > deadline:
            break
        
        # Find the row with the largest element
        max_row_idx = max(range(len(current_rows)), key=lambda i: current_rows[i][-1])
        max_val = current_rows[max_row_idx][-1]
        
        if max_val <= 1:
            break
        
        # Try to reduce the last element of this row
        # or swap some elements
        found = False
        for new_val in range(max_val - 1, 0, -1):
            if deadline and time.time() > deadline:
                break
            
            # Check if we can replace the last element with new_val
            # while keeping the row strictly increasing
            row = current_rows[max_row_idx]
            if new_val <= row[-2] if len(row) > 1 else False:
                continue
            
            # Check if the new differences are all unused
            old_last = row[-1]
            new_diffs = {new_val - a for a in row[:-1]}
            
            # Remove old differences involving old_last
            old_diffs = {old_last - a for a in row[:-1]}
            
            # Check if new_diffs intersect with (used - old_diffs)
            used = get_used_diffs(current_rows)
            used.difference_update(old_diffs)
            
            if not new_diffs & used:
                row[-1] = new_val
                current_max = max(max(r) for r in current_rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in current_rows]
                found = True
                break
        
        if not found:
            # Try other rows
            for ri in range(len(current_rows)):
                if ri == max_row_idx:
                    continue
                row = current_rows[ri]
                last = row[-1]
                for new_val in range(last - 1, 0, -1):
                    if deadline and time.time() > deadline:
                        break
                    if len(row) > 1 and new_val <= row[-2]:
                        continue
                    new_diffs = {new_val - a for a in row[:-1]}
                    old_diffs = {last - a for a in row[:-1]}
                    used = get_used_diffs(current_rows)
                    used.difference_update(old_diffs)
                    if not new_diffs & used:
                        row[-1] = new_val
                        current_max = max(max(r) for r in current_rows)
                        if current_max < best_max:
                            best_max = current_max
                            best_rows = [r[:] for r in current_rows]
                        found = True
                        break
                if found:
                    break
        
        if not found:
            break
    
    return best_rows

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    with all differences distinct across all rows, minimizing the largest entry."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with greedy
    rows = _greedy_construct(n_rows, k)
    current_max = max(max(row) for row in rows)
    
    # Try local search to improve
    rows = _local_search(rows, k, deadline)
    current_max = max(max(row) for row in rows)
    
    # Try backtracking with a tighter bound if time permits
    if time.time() < deadline - 10:
        bt_rows = _backtrack_find(n_rows, k, deadline)
        if bt_rows is not None:
            bt_max = max(max(row) for row in bt_rows)
            if bt_max < current_max:
                rows = bt_rows
                current_max = bt_max
    
    # Final local search
    if time.time() < deadline - 5:
        rows = _local_search(rows, k, deadline)
    
    return rows
# EVOLVE-BLOCK-END
