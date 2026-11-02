"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 115
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple strategies
    # Strategy 1: Greedy with different starting biases
    for seed in range(20):
        if time.time() >= deadline - 10:
            break
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    # Strategy 2: Local search from best
    if best_rows is not None:
        best_rows, best_scope = _local_search_fast(best_rows, n_rows, k, deadline)
    
    # Strategy 3: More aggressive search with larger moves
    if time.time() < deadline - 20:
        for seed in range(5):
            if time.time() >= deadline - 10:
                break
            rows = _greedy_construct(n_rows, k, seed + 100)
            if rows is not None:
                rows, scope = _local_search_fast(rows, n_rows, k, deadline)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
    
    return best_rows

def _greedy_construct(n_rows, k, seed=0):
    """Greedy construction trying to minimize scope."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
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

def _get_used_diffs(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used

def _is_valid(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    return max(max(row) for row in rows)

def _compute_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _local_search_fast(rows, n_rows, k, deadline):
    """Faster local search with precomputed differences."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    rng = random.Random(42)
    
    # Precompute differences for each row
    row_diffs = [_compute_diffs(r) for r in best_rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try moving one element in one row
        for _ in range(500):
            if time.time() >= deadline - 5:
                break
            
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            if len(row) <= 1:
                continue
            
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            
            # Try various deltas
            deltas = list(range(-5, 6)) + list(range(-10, 11, 2)) + [rng.randint(-20, 20)]
            rng.shuffle(deltas)
            
            for delta in deltas:
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                
                # Remove old differences for this row
                old_diffs = _compute_diffs(row)
                
                # Apply change
                row[pos] = new_val
                new_diffs = _compute_diffs(row)
                
                # Check if new diffs conflict with other rows
                other_diffs = all_diffs - old_diffs
                if new_diffs & other_diffs:
                    row[pos] = old_val
                    continue
                
                # Check for internal duplicates in the modified row
                expected_diff_count = len(row) * (len(row) - 1) // 2
                if len(new_diffs) != expected_diff_count:
                    row[pos] = old_val
                    continue

                # Valid! Update
                all_diffs = (all_diffs - old_diffs) | new_diffs
                new_scope = _scope(best_rows)
                
                if new_scope < best_scope:
                    best_scope = new_scope
                    row_diffs[ri] = new_diffs
                    improved = True
                    break
                else:
                    # Revert
                    row[pos] = old_val
                    all_diffs = (all_diffs - new_diffs) | old_diffs
                    row_diffs[ri] = old_diffs
            
            if improved:
                break
        
        if not improved:
            # Try rebuilding one row
            if time.time() >= deadline - 5:
                break
            
            ri = rng.randrange(n_rows)
            other_diffs = set()
            for j in range(n_rows):
                if j != ri:
                    other_diffs |= row_diffs[j]
            
            # Rebuild row greedily, trying to minimize max
            new_row = [0]
            x = 0
            while len(new_row) < k + 1 and time.time() < deadline - 5:
                x += 1
                new = {x - a for a in new_row}
                if not new & other_diffs:
                    new_row.append(x)
                    other_diffs |= new
            
            if len(new_row) == k + 1:
                candidate = [list(r) for r in best_rows]
                candidate[ri] = new_row
                new_scope = _scope(candidate)
                if new_scope <= best_scope:
                    best_rows = candidate
                    best_scope = new_scope
                    row_diffs[ri] = _compute_diffs(new_row)
                    all_diffs = set()
                    for rd in row_diffs:
                        all_diffs |= rd
    
    return best_rows, best_scope
# EVOLVE-BLOCK-END
