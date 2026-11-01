import random
import math
import time

def construct(n_rows, k):
    start_time = time.time()
    timeout = 115.0
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def randomized_greedy_row(forbidden_diffs, max_limit=None, max_candidates=5):
        row = [0]
        used_in_row = set()
        last_val = 0
        
        for _ in range(k):
            candidates = []
            # Look ahead a bit to find candidates
            # We need x such that x - a is not in forbidden and not in used_in_row
            # And x - a must be distinct for all a in row (implied if we check set size or just check intersection)
            # Actually, new_diffs = {x - a for a in row}. 
            # We need new_diffs & forbidden_diffs == empty
            # And new_diffs & used_in_row == empty
            # And len(new_diffs) == len(row) (distinctness within new diffs)
            
            # To speed up, we can iterate x from last_val + 1 upwards
            # But we want randomness.
            # Let's find the first few valid x's.
            
            x = last_val + 1
            while len(candidates) < max_candidates:
                # If max_limit is set, stop if x >= max_limit
                if max_limit is not None and x >= max_limit:
                    break
                    
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row) and len(new_diffs) == len(row):
                    candidates.append(x)
                
                x += 1
                # Safety break to avoid infinite loop if no candidates found early
                if x - last_val > 200: 
                    break

            if not candidates:
                return None
            
            # Pick randomly from candidates
            # Weight towards smaller values? Uniform is fine for diversity.
            chosen_x = random.choice(candidates)
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used_in_row |= new_diffs
            last_val = chosen_x
            
        return row

    # Initial solution
    rows = greedy_construct()
    all_diffs = [get_row_diffs(r) for r in rows]
    current_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = current_max
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.99
    
    while time.time() - start_time < timeout:
        if T < 0.01:
            T = 10.0 # Restart with high temp if stuck? Or just continue with low temp.
            # Better to just keep going, maybe increase T occasionally if no improvement for long.
            
        i = random.randint(0, n_rows - 1)
        
        # Calculate forbidden diffs for row i
        forbidden = set()
        for j in range(n_rows):
            if j != i:
                forbidden |= all_diffs[j]
        
        # Try to find a better row
        # Strategy 1: Try to find a row with max < current_max
        new_row = randomized_greedy_row(forbidden, max_limit=current_max, max_candidates=10)
        
        if new_row is None:
            # If we couldn't find one under current max, try to find any valid row
            # This might increase max, but could lead to better global structure later
            new_row = randomized_greedy_row(forbidden, max_limit=None, max_candidates=10)
            
        if new_row is None:
            continue
            
        new_max_row = new_row[-1]
        
        # Check if new row is valid with others (it is, by construction against forbidden)
        # But we need to check if it creates collisions? 
        # randomized_greedy_row ensures new_row's diffs don't intersect with forbidden (others).
        # And internally distinct.
        # So the new set is valid.
        
        new_global_max = max(max(r[-1] for r in rows if r is not new_row), new_max_row)
        # Note: rows[i] is being replaced.
        # We need to be careful with the max calculation.
        # Let's just compute max of all rows if we replace i.
        # Since n_rows is small (7), we can just compute it.
        
        # Actually, let's just swap and compute max.
        old_row = rows[i]
        rows[i] = new_row
        all_diffs[i] = get_row_diffs(new_row)
        
        new_global_max = max(r[-1] for r in rows)
        
        delta = new_global_max - current_max
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True
            
        if accept:
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = rows[:]
        else:
            # Reject: revert
            rows[i] = old_row
            all_diffs[i] = get_row_diffs(old_row)
            
        T *= cooling_rate
        
    return best_rows
