import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_rows = None
    best_max_val = float('inf')
    
    # Parameters for randomization
    # We look at the top 'window' valid candidates for the next element
    window_size = 10
    
    while time.time() < deadline:
        used_diffs = set()
        rows = []
        current_max_val = 0
        success = True
        
        for r in range(n_rows):
            row = [0]
            # We need k+1 elements, so we need k more
            for _ in range(k):
                last = row[-1]
                # Find valid next values
                candidates = []
                x = last + 1
                # We don't know how far to look, but we can stop if we found enough candidates
                # Or if x gets too large (heuristic: if x > current_max_val * 2, maybe stop? 
                # Actually, we just need to find SOME candidates. 
                # If we go too high, the solution is likely bad, but we still need a valid one to complete the row.
                # Let's limit the search for candidates to a reasonable bound to avoid hanging on a bad path.
                # If we can't find a candidate within last + 50, this path is likely bad, but we must complete it.
                # Let's just find the first 'window_size' valid candidates.
                
                while len(candidates) < window_size:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        candidates.append(x)
                    x += 1
                    # Safety: if x becomes huge, we might be in a dead end or very bad solution.
                    # But we must pick one.
                    if x > last + 200: # Arbitrary cap for candidate search in a single step
                        break
                
                if not candidates:
                    # This shouldn't happen unless we are stuck, but if we break early, we might have none.
                    # If none found, we must pick the smallest valid one we would have found?
                    # Actually, if we broke out of the while loop, we might have found some or none.
                    # If none found, we need to keep searching x indefinitely? 
                    # No, there are always valid x's eventually (infinite integers).
                    # So the loop above might exit if x > last+200 but candidates is empty.
                    # In that case, we must continue searching for a valid x.
                    if not candidates:
                        while True:
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used_diffs:
                                candidates.append(x)
                                break
                            x += 1
                
                # Pick a random candidate
                # To bias towards smaller values (which we want), we can pick from the first few
                # or use a weighted random choice.
                # Simple uniform from the window is a good start.
                next_val = random.choice(candidates)
                
                row.append(next_val)
                # Update used_diffs
                for a in row[:-1]:
                    used_diffs.add(next_val - a)
                
                if next_val > current_max_val:
                    current_max_val = next_val
            
            rows.append(row)
        
        if current_max_val < best_max_val:
            best_max_val = current_max_val
            best_rows = [list(r) for r in rows]
            
            # If we found a very good solution, we can stop early?
            # The theoretical lower bound is around 112. Best known is 134.
            # If we hit 134 or lower, we can stop.
            if best_max_val <= 134:
                break

    if best_rows is None:
        # Fallback to deterministic greedy if randomization failed completely (shouldn't happen)
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

    return best_rows
