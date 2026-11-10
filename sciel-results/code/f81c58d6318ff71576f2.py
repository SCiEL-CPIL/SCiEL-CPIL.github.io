"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Helper: compute all positive differences for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Helper: check if a full configuration is valid
    def is_valid(rows):
        used = set()
        for row in rows:
            rd = row_diffs(row)
            if rd & used:
                return False
            used |= rd
        return True
    
    # Helper: get max entry
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    # Generate a random valid configuration using greedy with randomization
    def random_greedy(n_rows, k, randomize_factor=0.0):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Build row greedily but with some randomization
            while len(row) < k + 1:
                # Try candidates in order with some randomization
                best_x = None
                candidates = []
                x = 1
                max_try = 200
                while len(candidates) < max_try:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        candidates.append(x)
                        if len(candidates) >= max_try:
                            break
                if not candidates:
                    # Fallback: keep searching
                    while True:
                        x += 1
                        new = {x - a for a in row}
                        if not (new & used):
                            candidates.append(x)
                            break
                
                if randomize_factor > 0 and len(candidates) > 1:
                    # Pick randomly from top candidates
                    idx = random.randint(0, min(len(candidates) - 1, int(randomize_factor * len(candidates))))
                    best_x = candidates[idx]
                else:
                    best_x = candidates[0]
                
                row.append(best_x)
                used |= {best_x - a for a in row[:-1]}
            rows.append(row)
        return rows
    
    # Local search move: try to reduce max entry by moving a mark
    def local_search(rows, n_rows, k, time_limit=10.0):
        best_rows = [list(r) for r in rows]
        best_max = max_entry(best_rows)
        current_rows = [list(r) for r in rows]
        current_max = max_entry(current_rows)
        
        start = time.time()
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < start + time_limit and no_improve < max_no_improve:
            # Pick a random row and a random non-first mark to try moving
            row_idx = random.randint(0, n_rows - 1)
            pos_idx = random.randint(1, k)  # positions 1..k (not 0)
            
            row = current_rows[row_idx]
            old_val = row[pos_idx]
            
            # Try to find a new value for this position that keeps row valid
            # and doesn't conflict with other rows' differences
            # First, compute differences of all other rows
            other_used = set()
            for i in range(n_rows):
                if i != row_idx:
                    other_used |= row_diffs(current_rows[i])
            
            # Differences within this row excluding the mark at pos_idx
            row_without = row[:pos_idx] + row[pos_idx+1:]
            row_without_diffs = row_diffs(row_without) if len(row_without) > 1 else set()
            
            # New value must create differences that don't conflict with:
            # 1. other rows' differences
            # 2. this row's other differences (row_without_diffs)
            # And the row must remain strictly increasing
            
            # Determine valid range for new value
            lo = row[pos_idx - 1] + 1 if pos_idx > 0 else 1
            hi = row[pos_idx + 1] - 1 if pos_idx < k else best_max + 50
            
            # Try to find a new value that reduces the max
            found = False
            # First try values that might reduce max
            if old_val == current_max and current_max == max(max(r) for r in current_rows):
                # Try to move this mark to a smaller value
                for new_val in range(lo, min(hi + 1, old_val)):
                    new_diffs = {new_val - a for a in row_without}
                    if new_diffs & other_used:
                        continue
                    if new_diffs & row_without_diffs:
                        continue
                    # Valid move
                    row[pos_idx] = new_val
                    new_max = max_entry(current_rows)
                    if new_max < current_max:
                        current_max = new_max
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(r) for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    found = True
                    break
            
            if not found:
                # Try a random valid move (for exploration)
                range_size = hi - lo + 1
                if range_size <= 0:
                    no_improve += 1
                    continue
                for _ in range(5):  # try a few random positions
                    new_val = random.randint(lo, hi)
                    if new_val == old_val:
                        continue
                    new_diffs = {new_val - a for a in row_without}
                    if new_diffs & other_used:
                        continue
                    if new_diffs & row_without_diffs:
                        continue
                    row[pos_idx] = new_val
                    new_max = max_entry(current_rows)
                    # Accept if it doesn't worsen max, or with small probability
                    if new_max <= current_max or random.random() < 0.05:
                        current_max = new_max
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(r) for r in current_rows]
                        no_improve = 0
                    else:
                        # Revert
                        row[pos_idx] = old_val
                    no_improve += 1
                    break
                else:
                    no_improve += 1
        
        return best_rows, best_max
    
    # Strategy: multiple restarts with local search
    best_rows = None
    best_max = float('inf')
    
    # First, get a baseline with greedy
    baseline = random_greedy(n_rows, k, randomize_factor=0.0)
    baseline_max = max_entry(baseline)
    best_rows = baseline
    best_max = baseline_max
    
    # Now do local search from this and other starting points
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        
        # Generate a diverse starting point
        if restart_count == 1:
            start_rows = random_greedy(n_rows, k, randomize_factor=0.0)
        elif restart_count % 3 == 0:
            start_rows = random_greedy(n_rows, k, randomize_factor=0.3)
        else:
            start_rows = random_greedy(n_rows, k, randomize_factor=0.1)
        
        start_max = max_entry(start_rows)
        
        # Do local search
        improved_rows, improved_max = local_search(start_rows, n_rows, k, time_limit=8.0)
        
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Final check: make sure we have a valid result
    if best_rows is None:
        best_rows = random_greedy(n_rows, k, randomize_factor=0.0)
    
    # Verify validity one more time
    if not is_valid(best_rows):
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows
    
    return best_rows
# EVOLVE-BLOCK-END
