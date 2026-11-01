"""Simulated annealing approach for difference triangle set."""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145.0  # 5s safety margin from 150s limit
    
    # Get a valid baseline using greedy
    baseline = _greedy_construct(n_rows, k)
    best_rows = baseline
    best_score = _max_entry(baseline)
    
    # For very small problems, return immediately
    if n_rows <= 1:
        return baseline
    
    # Simulated annealing
    current_rows, current_score = _simulated_annealing(n_rows, k, deadline)
    
    if current_score < best_score:
        best_score = current_score
        best_rows = current_rows
    
    # Verify validity before returning
    if not _is_valid(best_rows, n_rows, k):
        return baseline
    
    return best_rows

def _greedy_construct(n_rows, k):
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

def _simulated_annealing(n_rows, k, deadline):
    """
    Simulated annealing to minimize max entry while maintaining validity.
    """
    current_rows = _greedy_construct(n_rows, k)
    current_score = _max_entry(current_rows)
    
    best_rows = [row[:] for row in current_rows]
    best_score = current_score
    
    # Precompute row diffs
    row_diffs = [_get_diffs(row) for row in current_rows]
    
    # Global used differences set
    used_diffs = set()
    for rd in row_diffs:
        used_diffs |= rd
    
    # Annealing parameters
    T_start = 5.0
    T_end = 0.001
    
    step = 0
    max_steps = 1000000
    
    while time.time() < deadline - 3.0:
        progress = min(1.0, step / max_steps)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick random row and position
        row_idx = random.randint(0, n_rows - 1)
        pos_idx = random.randint(1, k)
        
        # Save original values for potential revert
        orig_row = current_rows[row_idx][:]
        orig_diffs = row_diffs[row_idx]
        orig_score = current_score
        
        # Pick random delta
        delta = random.choice([-3, -2, -1, 1, 2, 3, -5, 5, -8, 8, -12, 12, -20, 20])
        
        new_val = orig_row[pos_idx] + delta
        
        # Bounds check
        if new_val <= 0:
            step += 1
            continue
        if pos_idx > 0 and new_val <= orig_row[pos_idx - 1]:
            step += 1
            continue
        if pos_idx < k and new_val >= orig_row[pos_idx + 1]:
            step += 1
            continue
        
        # Build test row
        test_row = orig_row[:]
        test_row[pos_idx] = new_val
        
        # Check internal validity
        new_row_diffs = _get_diffs(test_row)
        expected = len(test_row) * (len(test_row) - 1) // 2
        if len(new_row_diffs) != expected:
            step += 1
            continue
        
        # Check cross-row conflicts
        # A diff d in new_row_diffs conflicts if d is in used_diffs and d not in orig_diffs
        # (because orig_diffs will be removed from used_diffs when we apply the move)
        conflict = False
        for d in new_row_diffs:
            if d in used_diffs and d not in orig_diffs:
                conflict = True
                break
        
        if conflict:
            step += 1
            continue
        
        # Move is valid! Apply it
        current_rows[row_idx] = test_row
        row_diffs[row_idx] = new_row_diffs
        
        # Update used_diffs
        used_diffs = set()
        for rd in row_diffs:
            used_diffs |= rd
        
        new_score = _max_entry(current_rows)
        
        # Acceptance
        if new_score <= orig_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows = [row[:] for row in current_rows]
        else:
            delta_cost = new_score - orig_score
            if T > 1e-10:
                prob = math.exp(-delta_cost / T)
            else:
                prob = 0.0
            if random.random() < prob:
                current_score = new_score
            else:
                # Revert
                current_rows[row_idx] = orig_row
                row_diffs[row_idx] = orig_diffs
                used_diffs = set()
                for rd in row_diffs:
                    used_diffs |= rd
                current_score = orig_score
        
        step += 1
    
    if best_score < current_score:
        return best_rows, best_score
    return current_rows, current_score

def _get_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)
# EVOLVE-BLOCK-END
