"""Difference triangle set (7,5) solver with Simulated Annealing and efficient diff tracking."""
import time
import random
import sys

# Deadline guard: return best result before 145s (5s safety margin under 150s kill)
_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    """Return the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy_row(used, k, rng, max_val=None):
    """Build one row greedily, choosing the smallest available next mark.
    used: set of forbidden differences.
    max_val: optional upper bound for entries.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            break
        # Occasionally jump ahead to create diversity
        if rng.random() < 0.1:
            x += rng.randint(1, 3)
            if max_val is not None and x > max_val:
                break
        
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _greedy_full(n_rows, k, rng, max_val=None):
    """Build all rows greedily with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_val)
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _sa_optimize(rows, k, rng, time_budget=60):
    """Simulated Annealing to minimize the scope (max entry)."""
    n_rows = len(rows)
    # Initial state
    best_rows = [list(r) for r in rows]
    current_rows = [list(r) for r in best_rows]
    
    # Track used differences globally for fast conflict checking
    # We need to be able to add/remove row diffs quickly.
    # Since k is small (5), each row has 15 diffs.
    # Total diffs is manageable.
    
    def get_diffs(row):
        return _row_diffs(row)

    # Initialize global diff counts
    diff_counts = {}
    for row in current_rows:
        for d in get_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
    def count_conflicts():
        c = 0
        for cnt in diff_counts.values():
            if cnt > 1:
                c += cnt - 1
        return c

    def get_scope():
        return _max_entry(current_rows)

    # Objective: Minimize scope + Penalty * Conflicts
    # We want valid solutions (0 conflicts) with low scope.
    # If invalid, we prioritize reducing conflicts first.
    
    current_scope = get_scope()
    current_conflicts = count_conflicts()
    current_cost = current_scope + 1000 * current_conflicts
    
    best_scope = current_scope
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    start_time = time.time()
    temp = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
            
        # Pick a random move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Move: Rebuild a random row
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            
            # Remove old row diffs from counts
            for d in get_diffs(old_row):
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Rebuild row with some randomness, trying to stay within current scope
            # Use a greedy approach with a slight bias towards smaller numbers
            used = set(diff_counts.keys())
            max_limit = current_scope + rng.randint(0, 5) # Allow slight expansion to escape
            new_row = _greedy_row(used, k, rng, max_limit)
            
            if len(new_row) < k + 1:
                # Restore old row
                for d in get_diffs(old_row):
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                continue
            
            # Add new row diffs
            for d in get_diffs(new_row):
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            current_rows[ri] = new_row
            
        else:
            # Move: Shift an entry in a random row
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if len(row) < 2:
                continue
            
            # Pick a random index > 0
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            
            # Try to reduce value to compress scope
            # Or randomly perturb
            if rng.random() < 0.6:
                # Try to reduce
                reduction = rng.randint(1, 3)
                new_val = old_val - reduction
                if new_val <= 0:
                    continue
                # Check strictly increasing
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
            else:
                # Random small change
                delta = rng.randint(-3, 3)
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
            
            # If new_val == old_val, skip
            if new_val == old_val:
                continue
                
            # Calculate diff changes
            old_diffs = get_diffs(row)
            
            # Remove old diffs
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Apply change
            row[idx] = new_val
            new_row = row
            new_diffs = get_diffs(new_row)
            
            # Add new diffs
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            # Check validity of the row itself (strictly increasing is already checked)
            # But we need to ensure no internal duplicates in the new row?
            # The problem says "increasing integers", so strictly increasing is required.
            # Our check `new_val <= row[idx-1]` etc ensures that.
            
            # Evaluate cost
            new_scope = get_scope()
            new_conflicts = count_conflicts()
            new_cost = new_scope + 1000 * new_conflicts
            
            delta_cost = new_cost - current_cost
            
            # Acceptance criterion
            if delta_cost < 0 or (temp > 0 and rng.random() < pow(2.718281828, -delta_cost / temp)):
                current_scope = new_scope
                current_conflicts = new_conflicts
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = current_scope
                    best_conflicts = current_conflicts
                    best_rows = [list(r) for r in current_rows]
            else:
                # Reject: Revert changes
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    
                row[idx] = old_val
                # current_rows[ri] is already updated to new_val, so revert it
                current_rows[ri][idx] = old_val

        # Cool down
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    # Return the best valid solution found, or the best overall if no valid found
    # The task requires a valid solution.
    # If best_conflicts > 0, we should try to return a valid one from history if possible,
    # or just return the best_rows and hope it's valid? No, it must be valid.
    # Let's check if best_rows is valid.
    if _is_valid(best_rows, k):
        return best_rows
    else:
        # Fallback: try to find any valid solution from the process?
        # Or just return current if valid?
        if _is_valid(current_rows, k):
            return current_rows
        # If neither is valid, return best_rows (which might be invalid, but we hope the greedy init was valid)
        return best_rows

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy restarts to find a valid start
    best_rows = None
    best_score = float('inf')
    
    n_restarts = 100
    for i in range(n_restarts):
        if _elapsed() > _DEADLINE - 20:
            break
        rng2 = random.Random(12345 + i * 7919)
        rows = _greedy_full(n_rows, k, rng2)
        if rows is None:
            continue
        if not _is_valid(rows, k):
            continue
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy without randomization jumps
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_score = _max_entry(best_rows)
    
    # Phase 2: Simulated Annealing
    if _elapsed() < _DEADLINE - 10:
        time_left = _DEADLINE - 10 - _elapsed()
        # Limit SA time to ensure we return in time
        sa_time = min(60, time_left)
        improved_rows = _sa_optimize(best_rows, k, rng, time_budget=sa_time)
        
        if improved_rows and _is_valid(improved_rows, k):
            new_score = _max_entry(improved_rows)
            if new_score < best_score:
                best_rows = improved_rows
                best_score = new_score
            else:
                # Keep original if SA didn't improve scope
                pass
    
    # Final validation
    if not _is_valid(best_rows, k):
        # Shouldn't happen, but fallback to a known valid greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    return best_rows
