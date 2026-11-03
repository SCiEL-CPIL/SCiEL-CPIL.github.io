"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin for 120s limit
    
    # 1. Greedy Initialization
    # Produces a valid set of rows with distinct differences.
    rows = greedy_init(n_rows, k)
    
    # Calculate initial state
    diff_counts = {}
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
    # Collision penalty: sum of (count - 1) for all counts > 1
    # Since greedy ensures distinctness, this should be 0 initially.
    def get_penalty(counts):
        pen = 0
        for c in counts.values():
            if c > 1:
                pen += c - 1
        return pen

    current_penalty = get_penalty(diff_counts)
    current_max = max(max(r) for r in rows)
    current_score = current_max + 10000 * current_penalty
    
    best_rows = [r[:] for r in rows]
    best_score = current_score
    
    # 2. Simulated Annealing
    T = 50.0  # Initial temperature
    cooling = 0.9995
    
    # Helper to get diffs for a specific row and index
    def get_row_diffs(row, idx):
        diffs = []
        # Diffs where idx is the larger index
        for i in range(idx):
            diffs.append(row[idx] - row[i])
        # Diffs where idx is the smaller index
        for j in range(idx + 1, len(row)):
            diffs.append(row[j] - row[idx])
        return diffs

    steps = 0
    while time.time() < deadline:
        # Select random move
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        # Choose a position to change. 
        # Changing 0 is not allowed (fixed at 0).
        # Changing the last element affects max.
        pos = random.randint(1, k) # 1 to k (0 is fixed)
        
        # Determine valid range for new value
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < k else current_max + 10 # Allow expansion? 
        # Actually, to minimize max, we might want to shrink.
        # But we also need to fix collisions.
        # Let's allow the value to move anywhere in (low, high) 
        # but also allow it to go higher if needed to fix collisions?
        # If we go higher, current_max might increase.
        # The score function handles the trade-off.
        
        # To allow shrinking the max, we should allow pos=k to go lower.
        # To allow fixing collisions, we might need to go higher temporarily.
        # Let's sample from a range that includes the current value and neighbors.
        
        if low >= high:
            continue
            
        # Sample new value
        # Bias towards current value?
        # Uniform in [low, high]
        new_val = random.randint(low, high)
        
        if new_val == row[pos]:
            continue
            
        # Calculate old diffs
        old_diffs = get_row_diffs(row, pos)
        
        # Remove old diffs from counts
        for d in old_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
                
        # Apply new value
        old_val = row[pos]
        row[pos] = new_val
        
        # Calculate new diffs
        new_diffs = get_row_diffs(row, pos)
        
        # Add new diffs to counts
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
        # Calculate new score
        new_max = max(max(r) for r in rows)
        new_penalty = get_penalty(diff_counts)
        new_score = new_max + 10000 * new_penalty
        
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_score = new_score
            current_max = new_max
            current_penalty = new_penalty
            
            if current_score < best_score:
                best_score = current_score
                best_rows = [r[:] for r in rows]
                
            # Cool down
            T *= cooling
        else:
            # Revert
            row[pos] = old_val
            
            # Revert counts: remove new, add old
            for d in new_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1

        steps += 1
        
    return best_rows

def greedy_init(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    break
                x += 1
            row.append(x)
            used.update(new_diffs)
        rows.append(row)
    return rows
# EVOLVE-BLOCK-END
