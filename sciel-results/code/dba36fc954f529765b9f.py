import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    # 1. Initial Greedy Construction
    rows, used = _greedy_construct(n_rows, k)
    
    if not rows:
        return []
        
    current_scope = max(max(row) for row in rows)
    
    # 2. Local Search with Re-insertion and Simulated Annealing
    # We try to improve the scope by re-inserting rows greedily.
    # To escape local minima, we allow temporary worsening with SA probability.
    
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    best_rows = [row[:] for row in rows]
    best_scope = current_scope
    
    while time.time() < deadline:
        # Pick a random row to re-insert
        r = random.randint(0, n_rows - 1)
        
        # Remove row r's differences from used
        old_row = rows[r]
        _remove_differences(old_row, used)
        
        # Reconstruct row r greedily
        new_row = _greedy_row(old_row[0], k, used) # old_row[0] is always 0
        
        # Check if new row is valid (it should be, but check just in case)
        # _greedy_row ensures no conflicts with 'used'
        
        # Calculate new scope
        # We need to know the max of the whole set.
        # Since we only changed row r, we can compute max of other rows + max of new_row
        # But for simplicity, just compute max of all rows
        new_scope = max(max(row) for row in rows[:r] + [new_row] + rows[r+1:])
        
        # Acceptance criterion
        if new_scope < current_scope:
            # Always accept improvement
            rows[r] = new_row
            _add_differences(new_row, used)
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [row[:] for row in rows]
        else:
            # Simulated Annealing: accept worse solution with probability exp(-delta/T)
            delta = new_scope - current_scope
            if delta > 0 and random.random() < math.exp(-delta / T):
                rows[r] = new_row
                _add_differences(new_row, used)
                current_scope = new_scope
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # If T is very low, we might want to restart or just stop if no improvement for long
        # For now, just keep going until time runs out.
        
    return best_rows

def _greedy_construct(n_rows, k):
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
    return rows, used

def _greedy_row(start_val, k, used):
    """
    Construct a row starting at start_val (which is 0) with k+1 elements.
    Uses the smallest available integers that don't conflict with 'used'.
    """
    row = [start_val]
    x = start_val
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _remove_differences(row, used):
    """Remove all differences from 'row' from the 'used' set."""
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diff = row[j] - row[i]
            if diff in used:
                used.remove(diff)

def _add_differences(row, used):
    """Add all differences from 'row' to the 'used' set."""
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diff = row[j] - row[i]
            used.add(diff)
# EVOLVE-BLOCK-END
