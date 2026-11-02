"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    
    # Helper to compute all differences for a row
    def get_diffs(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))]
    
    # 1. Greedy Initialization
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_scope = max(max(r) for r in best_rows)
    best_valid = True # Greedy is always valid
    
    # Data structures for SA
    # diff_count[d] = number of times difference d appears in the current configuration
    
    def build_diff_count(rows):
        counts = {}
        for r in rows:
            for d in get_diffs(r):
                counts[d] = counts.get(d, 0) + 1
        return counts

    current_rows = [r[:] for r in best_rows]
    current_diff_count = build_diff_count(current_rows)
    
    def get_collision_count(counts):
        # Number of excess occurrences
        return sum(c - 1 for c in counts.values() if c > 1)
    
    current_collisions = get_collision_count(current_diff_count)
    current_max = max(max(r) for r in current_rows)
    
    # Cost function: Max + Penalty * Collisions
    PENALTY = 1000
    
    def get_cost(max_val, collisions):
        return max_val + PENALTY * collisions

    current_cost = get_cost(current_max, current_collisions)
    
    # SA Parameters
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    max_steps = 2000000 # Limit steps to ensure we check time
    
    steps = 0
    while time.time() < deadline and steps < max_steps:
        if T < T_min:
            break
            
        # Pick a random row and index
        i = random.randint(0, n_rows - 1)
        j = random.randint(1, k) # Index 1 to k (0 is fixed at 0)
        
        old_val = current_rows[i][j]
        
        # Determine valid range for new value
        lower = current_rows[i][j-1] + 1
        if j < k:
            upper = current_rows[i][j+1] - 1
        else:
            # If it's the last element, allow some expansion but bias towards reduction
            # We cap it to current_max + 10 to avoid drifting too far
            upper = max(old_val, current_max) + 10
            
        if lower > upper:
            T *= cooling
            continue
            
        # Sample new value
        new_val = random.randint(lower, upper)
        
        if new_val == old_val:
            T *= cooling
            continue
            
        # Calculate change in differences
        # 1. Remove old diffs involving index j
        for idx in range(j):
            d = old_val - current_rows[i][idx]
            current_diff_count[d] -= 1
            if current_diff_count[d] == 0:
                del current_diff_count[d]
                
        for idx in range(j+1, k+1):
            d = current_rows[i][idx] - old_val
            current_diff_count[d] -= 1
            if current_diff_count[d] == 0:
                del current_diff_count[d]
        
        # 2. Apply new value
        current_rows[i][j] = new_val
        
        # 3. Add new diffs
        for idx in range(j):
            d = new_val - current_rows[i][idx]
            current_diff_count[d] = current_diff_count.get(d, 0) + 1
            
        for idx in range(j+1, k+1):
            d = current_rows[i][idx] - new_val
            current_diff_count[d] = current_diff_count.get(d, 0) + 1
            
        # 4. Calculate new metrics
        new_collisions = get_collision_count(current_diff_count)
        # Optimization: new_max calculation
        # If new_val > current_max, new_max = new_val
        # Else if old_val == current_max and new_val < old_val, we might have reduced max
        # Otherwise current_max stays same (unless old_val was max and we reduced it, but we don't track which row had max)
        # Simple recalculation is O(N*K) which is tiny.
        new_max = max(max(r) for r in current_rows)
        
        new_cost = get_cost(new_max, new_collisions)
        delta_cost = new_cost - current_cost
        
        # Acceptance
        if delta_cost < 0 or (T > 0 and random.random() < math.exp(-delta_cost / T)):
            current_cost = new_cost
            current_collisions = new_collisions
            current_max = new_max
            
            # Update best if valid and better
            if new_collisions == 0 and new_max < best_scope:
                best_scope = new_max
                best_rows = [r[:] for r in current_rows]
                best_valid = True
        else:
            # Reject: Revert changes
            # 1. Remove new diffs
            for idx in range(j):
                d = new_val - current_rows[i][idx]
                current_diff_count[d] -= 1
                if current_diff_count[d] == 0:
                    del current_diff_count[d]
            for idx in range(j+1, k+1):
                d = current_rows[i][idx] - new_val
                current_diff_count[d] -= 1
                if current_diff_count[d] == 0:
                    del current_diff_count[d]
            
            # 2. Revert value
            current_rows[i][j] = old_val
            
            # 3. Re-add old diffs
            for idx in range(j):
                d = old_val - current_rows[i][idx]
                current_diff_count[d] = current_diff_count.get(d, 0) + 1
            for idx in range(j+1, k+1):
                d = current_rows[i][idx] - old_val
                current_diff_count[d] = current_diff_count.get(d, 0) + 1
            
            # Metrics revert implicitly
        
        T *= cooling
        steps += 1
        
    # Local Search Polish
    # Try to reduce each mark greedily
    # We do this on the best_rows found so far
    if best_valid:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Rebuild diff count for best_rows
            best_diff_count = build_diff_count(best_rows)
            best_max = max(max(r) for r in best_rows)
            
            for i in range(n_rows):
                for j in range(k, 0, -1): # From end to start
                    # Try to reduce best_rows[i][j]
                    current_val = best_rows[i][j]
                    lower = best_rows[i][j-1] + 1
                    
                    # Remove current diffs
                    old_diffs = []
                    for idx in range(j):
                        d = current_val - best_rows[i][idx]
                        old_diffs.append(d)
                        best_diff_count[d] -= 1
                        if best_diff_count[d] == 0: del best_diff_count[d]
                    for idx in range(j+1, k+1):
                        d = best_rows[i][idx] - current_val
                        old_diffs.append(d)
                        best_diff_count[d] -= 1
                        if best_diff_count[d] == 0: del best_diff_count[d]
                    
                    found = False
                    # Try values strictly smaller than current_val
                    for v in range(lower, current_val):
                        # Check if v is valid
                        valid = True
                        new_diffs = []
                        for idx in range(j):
                            d = v - best_rows[i][idx]
                            new_diffs.append(d)
                            if best_diff_count.get(d, 0) > 0:
                                valid = False
                                break
                        if not valid: continue
                        for idx in range(j+1, k+1):
                            d = best_rows[i][idx] - v
                            new_diffs.append(d)
                            if best_diff_count.get(d, 0) > 0:
                                valid = False
                                break
                        
                        if valid:
                            # Accept v
                            best_rows[i][j] = v
                            for d in new_diffs:
                                best_diff_count[d] = best_diff_count.get(d, 0) + 1
                            improved = True
                            found = True
                            break
                            
                    if not found:
                        # Revert: add back old diffs
                        for d in old_diffs:
                            best_diff_count[d] = best_diff_count.get(d, 0) + 1
        
        # Final check
        final_max = max(max(r) for r in best_rows)
        if final_max < best_scope:
            best_scope = final_max
            
    return best_rows
# EVOLVE-BLOCK-END
