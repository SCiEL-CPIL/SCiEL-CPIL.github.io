import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Helper to check validity and get diffs
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_move(rows, used_diffs, r, j, new_val):
        # Check bounds
        if j > 0 and new_val <= rows[r][j - 1]:
            return False
        if j < k and new_val >= rows[r][j + 1]:
            return False
        
        # Calculate old diffs involving rows[r][j]
        old_diffs = set()
        for i in range(j):
            old_diffs.add(rows[r][j] - rows[r][i])
        for i in range(j + 1, k + 1):
            old_diffs.add(rows[r][i] - rows[r][j])
        
        # Calculate new diffs involving new_val
        new_diffs = set()
        for i in range(j):
            new_diffs.add(new_val - rows[r][i])
        for i in range(j + 1, k + 1):
            new_diffs.add(rows[r][i] - new_val)
        
        # Check conflicts
        # The new_diffs must not intersect with (used_diffs - old_diffs)
        # Also new_diffs must be internally distinct (they are, since row is strictly increasing)
        
        # Efficient check:
        # For d in new_diffs:
        #   if d in used_diffs and d not in old_diffs: return False
        
        # Optimization:
        # If len(new_diffs) is small, just check.
        for d in new_diffs:
            if d in used_diffs:
                if d not in old_diffs:
                    return False
        return True

    def apply_move(rows, used_diffs, r, j, new_val):
        old_val = rows[r][j]
        
        old_diffs = set()
        for i in range(j):
            old_diffs.add(old_val - rows[r][i])
        for i in range(j + 1, k + 1):
            old_diffs.add(rows[r][i] - old_val)
            
        new_diffs = set()
        for i in range(j):
            new_diffs.add(new_val - rows[r][i])
        for i in range(j + 1, k + 1):
            new_diffs.add(rows[r][i] - new_val)
            
        # Remove old, add new
        for d in old_diffs:
            used_diffs.discard(d)
        for d in new_diffs:
            used_diffs.add(d)
            
        rows[r][j] = new_val

    def revert_move(rows, used_diffs, r, j, new_val, old_val):
        # Swap back
        apply_move(rows, used_diffs, r, j, old_val)

    # Initial Solution: Randomized Greedy
    def generate_random_valid(max_scope_hint):
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            # We want to keep it small, but randomize
            # Estimate next max
            current_max = 0
            for _ in range(k):
                # Try to find a valid x > current_max
                # Randomize the search range
                # Start from current_max + 1, go up to current_max + 50 (or similar)
                found = False
                # Sort candidates by distance to current_max? 
                # Just pick random candidates in a window
                window_size = 20
                candidates = []
                # We need to ensure we don't miss small values
                # Let's try values from current_max + 1 to current_max + window_size
                # But we must check validity
                # To be faster, we can just iterate and pick randomly from valid ones found
                # Or just iterate and pick the first valid one, but add randomness to the order?
                
                # Let's try a few random offsets
                # Actually, simple greedy is fast. Let's use randomized greedy:
                # Pick a random valid x from [current_max + 1, current_max + 10]
                # If none, expand range.
                
                valid_candidates = []
                # Check a range
                search_limit = current_max + 30
                for x in range(current_max + 1, search_limit + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(x)
                        
                if not valid_candidates:
                    # Expand search
                    for x in range(search_limit + 1, current_max + 100):
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            valid_candidates.append(x)
                            if len(valid_candidates) > 5: break
                            
                if not valid_candidates:
                    # Fallback to linear search
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            valid_candidates.append(x)
                            break
                        x += 1
                            
                # Pick a random candidate, biased towards smaller
                # Weighted random choice
                # Simpler: pick randomly from the first few smallest
                if len(valid_candidates) > 1:
                    # Sort to ensure we pick small ones with higher prob
                    valid_candidates.sort()
                    # Pick from first half
                    pick_range = valid_candidates[:max(1, len(valid_candidates)//2)]
                    x = random.choice(pick_range)
                else:
                    x = valid_candidates[0]
                    
                row.append(x)
                used |= {x - a for a in row[:-1]}
                current_max = x
            rows.append(row)
        return rows, used

    # Run multiple restarts
    best_rows = None
    best_scope = float('inf')
    
    num_restarts = 2000 # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial
        rows, used = generate_random_valid(150)
        current_scope = max(max(row) for row in rows)
        
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [row[:] for row in rows]
            
        # Local Search
        local_improvements = 0
        steps_since_improvement = 0
        max_steps_no_improve = 5000
        
        while time.time() < deadline:
            if steps_since_improvement > max_steps_no_improve:
                break
                
            # Pick random row and index
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k) # 1 to k (0 is fixed)
            
            # Determine range for new value
            low = rows[r][j-1] + 1
            high = rows[r][j+1] - 1 if j < k else rows[r][j] + 10 # Allow expansion if last
            
            # Bias towards reducing max
            if j == k:
                # Prefer smaller values
                # Sample from [low, min(high, rows[r][j] - 1)] with higher prob
                # Or just sample from [low, rows[r][j]]
                upper_bound = rows[r][j]
                if low > upper_bound:
                    continue
                new_val = random.randint(low, upper_bound)
            else:
                # Normal move
                if low > high:
                    continue
                new_val = random.randint(low, high)
                
            if new_val == rows[r][j]:
                continue
                
            if is_valid_move(rows, used, r, j, new_val):
                old_val = rows[r][j]
                apply_move(rows, used, r, j, new_val)
                
                new_scope = max(max(row) for row in rows)
                
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [row[:] for row in rows]
                    local_improvements += 1
                    steps_since_improvement = 0
                else:
                    # Accept non-improving moves to escape local minima?
                    # Simulated annealing style
                    # Or just accept if it doesn't increase max too much?
                    # For now, accept all valid moves to explore.
                    # This is hill climbing if we only accept improvements.
                    # Let's accept all valid moves to allow exploration.
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1
                
            # Periodic check for best
            # (Already done inside if)
            
        # End of local search for this restart

    if best_rows is None:
        # Fallback to simple greedy
        rows, _ = generate_random_valid(200)
        return rows
        
    return best_rows

# EVOLVE-BLOCK-START
# (The code above is the full program, but we need to wrap it in the function construct)
# Wait, the prompt asks for the COMPLETE program in a single code block.
# I will structure it properly.
# EVOLVE-BLOCK-END
