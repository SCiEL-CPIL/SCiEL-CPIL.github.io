import random
import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    best_rows = None
    best_max_val = float('inf')
    
    def get_max_val(rows):
        return max(max(row) for row in rows) if rows else 0

    def randomized_greedy_with_state():
        used_diffs = set()
        rows = []
        # Smaller window forces tighter packing initially
        window_size = 3 
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                candidates = []
                x = row[-1] + 1
                # Scan for candidates
                # We can limit the scan depth to avoid huge jumps if we are trying to stay small,
                # but for the first few rows, we might need to go further.
                # Let's scan up to 50 or until we have enough candidates.
                scan_limit = 100
                while len(candidates) < window_size and x < row[-1] + scan_limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Fallback: find the very first valid one
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                            break
                        x += 1
                            
                # Bias towards smaller values to keep scope tight
                # Use a weighted choice or just pick the smallest with some probability?
                # Let's pick randomly from the first few, but maybe sort them (they are already sorted)
                # and pick from the first half?
                # For now, random choice from candidates is okay, but let's try picking the smallest 
                # with 50% probability to encourage tightness.
                if len(candidates) > 1 and random.random() < 0.5:
                    chosen_x = candidates[0]
                else:
                    chosen_x = random.choice(candidates)
                    
                row.append(chosen_x)
                used_diffs.update({chosen_x - a for a in row[:-1]})
            
            rows.append(row)
        return rows, used_diffs

    def local_improve_with_state(rows, used_diffs):
        max_improve_iters = 200
        no_improve_count = 0
        
        for _ in range(max_improve_iters):
            if time.time() - start_time > time_limit - 2:
                break
                
            improved = False
            
            # Strategy 1: Try to shrink the largest element of the row with the current global max
            max_val = max(max(r) for r in rows)
            # Find all rows with this max
            target_indices = [i for i, r in enumerate(rows) if max(r) == max_val]
            
            # Try to improve one of them
            random.shuffle(target_indices)
            success_in_this_phase = False
            
            for target_idx in target_indices:
                target_row = rows[target_idx]
                last_elem = target_row[-1]
                
                # Try to replace last_elem with a smaller x
                lower_bound = target_row[-2] + 1
                upper_bound = last_elem - 1
                
                if lower_bound > upper_bound:
                    continue # Can't shrink last element further without breaking order or range
                
                old_diffs = {last_elem - a for a in target_row[:-1]}
                
                # We want the smallest valid x to minimize the row max
                found = False
                for x in range(lower_bound, upper_bound + 1):
                    new_diffs = {x - a for a in target_row[:-1]}
                    
                    conflict = False
                    for d in new_diffs:
                        if d in used_diffs and d not in old_diffs:
                            conflict = True
                            break
                    
                    if not conflict:
                        # Valid!
                        used_diffs -= old_diffs
                        used_diffs |= new_diffs
                        target_row[-1] = x
                        improved = True
                        success_in_this_phase = True
                        break
            
            if success_in_this_phase:
                no_improve_count = 0
                continue # Restart loop with new state
                
            # Strategy 2: If we couldn't shrink the max row, try "Re-insertion" of a random row
            # This helps escape local minima where the max row is stuck but another row
            # could be rearranged to free up space for the max row to shrink.
            
            if no_improve_count > 10:
                # Pick a random row to remove and re-insert
                target_idx = random.randint(0, n_rows - 1)
                target_row = rows[target_idx]
                
                # Remove from used_diffs
                row_diffs = set()
                for i in range(len(target_row)):
                    for j in range(i+1, len(target_row)):
                        row_diffs.add(target_row[j] - target_row[i])
                used_diffs -= row_diffs
                
                # Remove from rows
                rows.pop(target_idx)
                
                # Re-insert using greedy, trying to keep values low
                # We can try to insert it at the end (simplest)
                new_row = [0]
                # Limit the search to slightly above the current global max to encourage compression?
                # Or just standard greedy.
                while len(new_row) < k + 1:
                    candidates = []
                    x = new_row[-1] + 1
                    # Scan a bit
                    for _ in range(50):
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                        x += 1
                        if len(candidates) >= 5: break
                    
                    if not candidates:
                        x = new_row[-1] + 1
                        while True:
                            new_diffs = {x - a for a in new_row}
                            if not (new_diffs & used_diffs):
                                candidates.append(x)
                                break
                            x += 1
                            
                    chosen_x = random.choice(candidates)
                    new_row.append(chosen_x)
                    used_diffs.update({chosen_x - a for a in new_row[:-1]})
                
                rows.append(new_row)
                improved = True # We changed the state, even if max didn't drop immediately
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If we are stuck for a long time, break to allow a new restart
                if no_improve_count > 50:
                    break

        return rows, used_diffs

    def solve():
        best_rows = None
        best_score = float('inf')
        
        # Run multiple restarts
        while time.time() - start_time < time_limit:
            # 1. Generate
            rows, used_diffs = randomized_greedy_with_state()
            
            # 2. Improve
            rows, used_diffs = local_improve_with_state(rows, used_diffs)
            
            # 3. Evaluate
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [r[:] for r in rows]
                
            # If we are very close to the theoretical bound or known best, we can stop early?
            # The bound is 105. If we hit 105, we stop.
            if best_score <= 105:
                break

        return best_rows

    return solve()
