import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Generate initial greedy solution
    # This provides a valid starting point.
    rows = _greedy_construct(n_rows, k)
    if not rows:
        return []
    
    # Calculate initial best score
    best_rows = [r[:] for r in rows]
    best_score = _get_scope(best_rows)
    
    # 2. Simulated Annealing / Hill Climbing with Row Rebuilds
    # We will try to improve the scope.
    # State: list of rows.
    # Score: max element.
    # Move: Rebuild a random row greedily given the current global used differences.
    
    current_rows = [r[:] for r in rows]
    current_score = best_score
    
    # Helper to check validity and get used set
    def _get_used_set(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used.add(r[j] - r[i])
        return used

    # Current used set
    global_used = _get_used_set(current_rows)
    
    # Pre-define the rebuild function using closure for k
    def _try_rebuild_row(row_idx, used_other_diffs):
        """
        Attempts to construct a new row for position row_idx,
        avoiding conflicts with `used_other_diffs`.
        Returns (new_row, new_diffs).
        """
        new_row = [0]
        new_diffs = set()
        x = 0
        
        while len(new_row) < k + 1:
            x += 1
            
            candidate_diffs = {x - a for a in new_row}
            
            # Check internal consistency (diffs within this new row)
            if candidate_diffs & new_diffs:
                continue
            
            # Check global consistency (diffs from other rows)
            if candidate_diffs & used_other_diffs:
                continue
            
            # Accept x
            new_row.append(x)
            new_diffs |= candidate_diffs
            
        return new_row, new_diffs

    # Main Loop
    iterations = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random row to rebuild
        idx = random.randint(0, n_rows - 1)
        
        # Calculate used diffs of all OTHER rows
        # We maintain global_used.
        # 1. Remove diffs of row[idx]
        old_row = current_rows[idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        # Subtract from global_used
        for d in old_diffs:
            global_used.discard(d)
            
        # Rebuild row
        new_row, new_diffs = _try_rebuild_row(idx, global_used)
        
        # Check if new row is better (smaller max element)
        new_max = max(new_row)
        old_max = max(old_row)
        
        # We accept if new_max <= old_max.
        # This allows shuffling with same max, and improving with smaller max.
        # We reject if new_max > old_max to avoid increasing scope.
        
        if new_max <= old_max:
            # Accept
            # Update global_used
            global_used |= new_diffs
            current_rows[idx] = new_row
            
            # Update best if improved
            new_curr_global_max = max(max(r) for r in current_rows)
            if new_curr_global_max < best_score:
                best_score = new_curr_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Reject, revert global_used (we discarded old, so we must re-add old)
            global_used |= old_diffs
            no_improve_count += 1
            
        # If we have been stuck for a very long time, we might want to do something else.
        # But random row selection should suffice.
        
        # Safety check on time
        if time.time() > deadline - 5.0:
            break

    return best_rows

def _greedy_construct(n_rows, k):
    """
    Generate an initial valid set of rows using greedy method.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        row_diffs = set()
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and not (new_diffs & row_diffs):
                row.append(x)
                row_diffs |= new_diffs
                used |= new_diffs
        rows.append(row)
    return rows

def _get_scope(rows):
    if not rows:
        return 0
    return max(max(r) for r in rows)

# EVOLVE-BLOCK-END
