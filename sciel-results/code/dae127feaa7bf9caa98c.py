import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1  # Lower bound on scope

    # State representation
    # rows: list of lists of ints
    # used_diffs: set of ints (for fast lookup)
    # We need to update used_diffs efficiently.
    # Given small scope (~120), a boolean array of size 200 is faster than a set?
    # Let's use a set for simplicity first, or a numpy array for speed.
    MAX_SCOPE_EST = 200
    diff_used = np.zeros(MAX_SCOPE_EST + 1, dtype=bool)
    
    # Helper to calculate scope
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    # Helper to calculate conflicts count
    # This is expensive if done from scratch. We should maintain it.
    # But for SA, we often just need to evaluate the change.
    
    def count_conflicts(rows):
        """
        Returns the number of duplicate differences.
        """
        seen = set()
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        conflicts += 1
                    else:
                        seen.add(d)
        return conflicts

    def calculate_cost(rows):
        """
        Cost = Conflicts * 10000 + Scope
        """
        conf = count_conflicts(rows)
        scope = get_scope(rows)
        return conf * 10000 + scope

    # Initial Construction: Greedy
    def greedy_construct(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last = row[-1]
                # Try to find next element
                found = False
                # Bias towards compact
                for x in range(last + 1, lb + 50):
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    # Check if x is within reasonable bounds
                    if x > 150: break
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    # Fallback: try wider
                    for x in range(last + 1, 200):
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    best_rows = greedy_construct(42)
    if best_rows is None:
        # Fallback if greedy fails (shouldn't happen for small k)
        best_rows = [[0, 1, 2, 3, 4, 5, 6, 7] for _ in range(n_rows)] # Invalid, but placeholder
        # Actually, let's just try multiple seeds
        for s in range(100):
            best_rows = greedy_construct(s)
            if best_rows: break

    best_cost = calculate_cost(best_rows)
    best_scope = get_scope(best_rows)
    
    # Simulated Annealing
    current_rows = [list(r) for r in best_rows]
    current_cost = best_cost
    
    T_start = 100.0
    T_end = 0.1
    alpha = 0.999  # Cooling rate
    T = T_start
    
    steps = 0
    max_steps = 500000
    
    # Precompute differences for fast update? 
    # For now, recalculate cost on accept.
    
    while T > T_end and steps < max_steps:
        if time.time() > deadline - 5:
            break
            
        # Choose move type
        if random.random() < 0.1:
            # Move: Regenerate a random row (Global Repair/Jump)
            row_idx = random.randint(0, n_rows - 1)
            # Remove row
            old_row = current_rows.pop(row_idx)
            
            # Calculate used diffs from other rows
            used_by_others = set()
            for r in current_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        used_by_others.add(r[j] - r[i])
            
            # Greedily construct new row avoiding used_by_others
            new_row = [0]
            for _ in range(k):
                last = new_row[-1]
                found = False
                # Try compact first
                for x in range(last + 1, 150):
                    new_diffs = [x - a for a in new_row]
                    if any(d in used_by_others for d in new_diffs):
                        continue
                    # Also ensure internal diffs are distinct (they are, since x > last > ... > 0)
                    new_row.append(x)
                    found = True
                    break
                if not found:
                    # Try wider
                    for x in range(last + 1, 200):
                        new_diffs = [x - a for a in new_row]
                        if any(d in used_by_others for d in new_diffs):
                            continue
                        new_row.append(x)
                        found = True
                        break
                if not found:
                    # If we can't construct, restore old row
                    current_rows.insert(row_idx, old_row)
                    break
            else:
                # Successfully constructed
                current_rows.insert(row_idx, new_row)
        else:
            # Move: Shift a random element in a random row
            row_idx = random.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            if len(row) <= 1:
                T *= alpha
                steps += 1
                continue
                
            elem_idx = random.randint(1, len(row) - 1) # Don't move 0
            old_val = row[elem_idx]
            
            # Shift by random delta
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            
            # Ensure strict increase
            if elem_idx > 0 and new_val <= row[elem_idx - 1]:
                continue
            if elem_idx < len(row) - 1 and new_val >= row[elem_idx + 1]:
                continue
            if new_val <= 0:
                continue
                
            row[elem_idx] = new_val
            
        # Evaluate new state
        new_cost = calculate_cost(current_rows)
        delta_cost = new_cost - current_cost
        
        # Acceptance criterion
        if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
            current_cost = new_cost
            new_scope = get_scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Reject: revert
            if 'old_row' in locals() and 'row_idx' in locals() and random.random() < 0.1:
                # This logic is flawed if we don't track exactly what changed.
                # It's safer to store the old state before mutation.
                pass
        
        # The rejection logic above is buggy because I didn't store the exact previous state for simple shifts.
        # Let's restart the loop with a cleaner structure.
        
        T *= alpha
        steps += 1
        
    return best_rows

# The above draft has bugs in the SA loop (rejection logic). 
# I will rewrite the SA part properly in the final code.
