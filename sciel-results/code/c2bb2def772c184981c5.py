import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct over all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Helper to check if a set of new diffs conflicts with used_diffs
    def check_conflicts(new_diffs, used_diffs):
        return bool(new_diffs & used_diffs)
    
    def get_diffs_for_row(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # 1. Randomized Greedy Construction
    def greedy_randomized(n_rows, k, window_size=20, rand_prob=0.1):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Find valid candidates
                # Start from row[-1] + 1
                start_val = row[-1] + 1
                # We look for the smallest valid x, but sometimes pick a larger one
                valid_candidates = []
                # Limit search depth to avoid infinite loops, though greedy usually finds one quickly
                # A bound on x: maybe row[-1] + k + 10? 
                # Actually, just scan upwards.
                found_any = False
                for x in range(start_val, start_val + window_size * 5): # Look ahead
                    new = {x - a for a in row}
                    if not (new & used):
                        valid_candidates.append(x)
                        if len(valid_candidates) >= window_size:
                            break
                if not valid_candidates:
                    # Fallback to standard greedy if window search failed (shouldn't happen often)
                    x = start_val
                    while True:
                        new = {x - a for a in row}
                        if not (new & used):
                            valid_candidates.append(x)
                            break
                        x += 1
                
                if valid_candidates:
                    if random.random() < rand_prob and len(valid_candidates) > 1:
                        # Pick randomly from the valid candidates found
                        x = random.choice(valid_candidates)
                    else:
                        # Pick the smallest
                        x = valid_candidates[0]
                    
                    row.append(x)
                    used.update({x - a for a in row[:-1]}) # Note: row[:-1] is the previous state
                    # Wait, `row` was appended to? No, we check before appending.
                    # Let's re-verify the logic.
                    # `new` was calculated against the current `row` (before x).
                    # So we should update `used` with `new`.
                    used.update(new)
                else:
                    # Should not happen
                    break
            rows.append(row)
        return rows, used

    # 2. Local Optimization
    def local_optimize(rows, used, deadline):
        # Calculate initial max
        current_max = max(max(r) for r in rows)
        
        # Pre-compute diffs for each row to make removal faster?
        # For n_rows=4, k=7, rows have 8 elements. Diffs per row = 28. Total 112 diffs.
        # Sets are small. Recalculating is fine.
        
        while time.time() < deadline:
            # Pick a random row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            
            # Pick a random position to try to reduce
            # Focus on the end of the row to reduce scope directly
            # Or any position to create slack
            # Let's pick a position biased towards the end
            pos = random.randint(1, len(row) - 1)
            
            current_val = row[pos]
            
            # Try to find a smaller valid value for row[pos]
            # Lower bound: row[pos-1] + 1
            # Upper bound: current_val - 1
            # Also must be < row[pos+1] if pos < len(row)-1
            
            lower_bound = row[pos - 1] + 1
            upper_bound = current_val - 1
            
            if pos < len(row) - 1:
                upper_bound = min(upper_bound, row[pos + 1] - 1)
                
            if lower_bound > upper_bound:
                continue
                
            # We want to try smaller values first to improve score
            # But we must ensure no conflicts.
            # Let's collect the diffs that will be removed
            old_diffs = set()
            for i in range(len(row)):
                if i != pos:
                    if row[pos] > row[i]:
                        old_diffs.add(row[pos] - row[i])
                    else:
                        old_diffs.add(row[i] - row[pos])
            
            # Remove them from global used set temporarily?
            # Or just check against (used - old_diffs)
            available_diffs = used - old_diffs
            
            new_val_found = False
            # Iterate from lower_bound to upper_bound
            # We prefer smaller values
            for v in range(lower_bound, upper_bound + 1):
                # Calculate new diffs
                new_diffs = set()
                for i in range(len(row)):
                    if i != pos:
                        if v > row[i]:
                            new_diffs.add(v - row[i])
                        else:
                            new_diffs.add(row[i] - v)
                
                # Check if new_diffs conflicts with available_diffs
                if not (new_diffs & available_diffs):
                    # Valid move!
                    # Update row
                    row[pos] = v
                    # Update used set
                    used -= old_diffs
                    used |= new_diffs
                    
                    # Check if max decreased
                    new_max = max(max(r) for r in rows)
                    if new_max < current_max:
                        current_max = new_max
                    new_val_found = True
                    break
            # If we found a smaller value, we improved the local structure.
            # We continue the loop to try more.
            
        return rows, current_max

    best_rows = None
    best_score = float('inf')
    
    # Run multiple randomized greedy starts
    # Adjust parameters based on time
    attempts = 0
    while time.time() < deadline - 10: # Leave 10s for optimization of best
        # Randomize parameters slightly
        window = random.randint(10, 50)
        prob = random.uniform(0.0, 0.3)
        
        rows, used = greedy_randomized(n_rows, k, window, prob)
        
        # Check validity (should be valid by construction)
        # Score
        score = max(max(r) for r in rows)
        
        if score < best_score:
            best_score = score
            best_rows = [r[:] for r in rows]
            best_used = set(used)
            
            # If we found a very good score, we can spend more time optimizing it
            # But let's just keep searching for a better start first.
        
        attempts += 1
        # If we have time, we can optimize the current best
        # But let's finish the greedy phase first to find a good baseline.
        # If we've done many attempts, maybe stop greedy and start optimizing best.
        if attempts > 20 and time.time() > start_time + 30:
             break

    if best_rows is None:
        # Fallback to simple greedy if randomized failed or timed out immediately
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not (new & used):
                        break
                    x += 1
                row.append(x)
                used |= new
            rows.append(row)
        best_rows = rows
        best_used = used
        best_score = max(max(r) for r in rows)

    # Optimize the best solution found
    # We need to pass a copy to local_optimize to avoid modifying best_rows if it fails?
    # local_optimize modifies in place. If it crashes, we lose best_rows.
    # Let's copy.
    opt_rows = [r[:] for r in best_rows]
    opt_used = set(best_used)
    
    try:
        opt_rows, final_score = local_optimize(opt_rows, opt_used, deadline)
        if final_score < best_score:
            best_rows = opt_rows
            best_score = final_score
    except Exception:
        pass # Keep the unoptimized best if optimization fails

    return best_rows
