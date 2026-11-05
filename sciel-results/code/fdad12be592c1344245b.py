import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    best_rows = None
    best_max_val = float('inf')
    
    # Lower bound on max value is at least k * (k+1) / 2 ? No.
    # Counting bound: n_rows * k*(k+1)/2 <= M
    # 7 * 15 = 105.
    
    for restart in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        random.seed() # Or use a counter for reproducibility? No, random is fine.
        
        used_diffs = set()
        rows = []
        current_max = 0
        success = True
        
        for r in range(n_rows):
            row = [0]
            # We need to pick k more numbers
            for pos in range(k):
                valid_candidates = []
                # Start from the smallest possible next number
                # We can limit the search range to improve speed
                # If current_max is already high, we might want to keep it low
                # Let's search up to a dynamic limit
                # Initial limit: best_max_val - 1 if we have a solution, else some high number
                search_limit = best_max_val - 1 if best_max_val < float('inf') else 200
                
                # If we haven't found a good solution yet, 200 is safe
                # We can also tighten this:
                # The remaining positions need at least 1 gap each.
                # So x + (k - pos) <= search_limit
                max_x = search_limit - (k - pos)
                
                for x in range(row[-1] + 1, max_x + 1):
                    new_diffs = {x - a for a in row}
                    # Check against used_diffs
                    if new_diffs & used_diffs:
                        continue
                    # Check against current row diffs
                    # The diffs in the row are {b - a for a, b in row, a < b}
                    # We can maintain this set incrementally
                    # For simplicity, let's just check if any new diff is in the row's existing diffs
                    # Row diffs so far:
                    row_diffs = set()
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            row_diffs.add(row[j] - row[i])
                            
                    if new_diffs & row_diffs:
                        continue
                    
                    valid_candidates.append(x)
                    
                    # To speed up, we can break early if we have enough candidates
                    if len(valid_candidates) > 10:
                        break
                
                if not valid_candidates:
                    success = False
                    break
                
                # Pick a candidate
                # Bias towards smaller values
                # Pick uniformly from the first few
                pick_range = min(len(valid_candidates), 5)
                x = random.choice(valid_candidates[:pick_range])
                
                row.append(x)
                # Update used_diffs
                used_diffs.update({x - a for a in row[:-1]})
                # Note: We don't add diffs between new x and itself.
                # The loop above `x - a for a in row` where row is the OLD row is correct.
                # Wait, `row` is updated after the loop.
                # Inside the loop, `row` is the state before adding x.
                # So `x - a` for `a in row` are the new diffs.
                # Correct.
                
            if not success:
                break
            
            rows.append(row)
            current_max = max(current_max, max(row))
            
        if success and current_max < best_max_val:
            best_max_val = current_max
            best_rows = rows
            
            # If we reached the theoretical lower bound, we can stop
            if best_max_val == 105:
                break
                
    if best_rows is None:
        # Fallback to simple greedy if no solution found (should not happen)
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
