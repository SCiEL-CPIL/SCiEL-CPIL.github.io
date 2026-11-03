import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Helper to calculate max entry
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(r) for r in rows)

    # Helper to check if a specific row is valid given a set of used diffs from OTHER rows
    # Returns None if invalid, else set of diffs in this row
    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return None # Internal conflict
                diffs.add(d)
        return diffs

    # Greedy construction of a single row given used diffs and a max_limit
    def build_row_greedy(used, k, max_limit, randomness=0.5):
        row = [0]
        current_max = 0
        for _ in range(k):
            candidates = []
            # Look ahead a bit
            # We want to find x > row[-1] such that {x-a for a in row} is disjoint from used
            # And x <= max_limit
            
            # To speed up, we can start from row[-1] + 1 and go up to max_limit
            # But if max_limit is tight, we should be careful.
            
            last = row[-1]
            # We'll collect a few valid candidates to randomize among
            # This helps avoid getting stuck in the "smallest possible" trap immediately
            limit_for_candidates = min(max_limit, last + 50)
            
            x = last + 1
            while x <= limit_for_candidates:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append(x)
                x += 1
            
            if not candidates:
                # Try further out if we didn't find any close by, up to max_limit
                x = limit_for_candidates + 1
                while x <= max_limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
            
            if not candidates:
                return None
            
            # Randomize choice
            # With high probability, pick the smallest to keep scope small
            # With low probability, pick a random one to explore
            if random.random() < randomness:
                # Pick from the first few to keep it somewhat compact
                pool = candidates[:min(5, len(candidates))]
                chosen = random.choice(pool)
            else:
                chosen = candidates[0]
            
            row.append(chosen)
            # Update used locally? No, we pass 'used' as external context.
            # We need to update the local 'used' for subsequent steps in this row?
            # Yes, the row itself must have distinct diffs.
            # So we add this row's diffs to a temporary set for this row's construction.
            # Wait, the 'used' passed in is from OTHER rows.
            # We need to ensure new diffs don't conflict with 'used' AND don't conflict with previous diffs in THIS row.
            # The check `if not (new_diffs & used)` handles external conflicts.
            # We also need to ensure `new_diffs` doesn't conflict with diffs already in `row`.
            # Let's maintain a local set `row_used_diffs`.
            
            # Correction: The above logic didn't track internal row diffs properly in the candidate selection loop 
            # if we just checked against global 'used'.
            # Let's restart build_row_greedy to be correct.
            pass
            
        return None # Placeholder, will be rewritten below

    # Correct Greedy Row Builder
    def build_row_greedy_correct(used_global, k, max_limit, randomness=0.5):
        row = [0]
        row_diffs = set()
        
        for _ in range(k):
            last = row[-1]
            candidates = []
            
            # Search for valid x
            # We want to find x such that:
            # 1. x > last
            # 2. x <= max_limit
            # 3. {x - a for a in row} is disjoint from used_global
            # 4. {x - a for a in row} is disjoint from row_diffs
            # 5. {x - a for a in row} has no internal duplicates (implied by distinct a's and strict increase? 
            #    No, x-a1 == x-a2 implies a1==a2, so internal diffs in a single step are unique.
            #    But we must check against existing row_diffs)
            
            # To optimize, we can break early if we find enough candidates
            # But for strict minimization, we usually want the smallest.
            # Randomness allows exploring larger x to potentially free up smaller x for later rows?
            # Actually, for a single row, smaller is generally better for scope.
            
            x = last + 1
            found_any = False
            # Limit search depth to avoid wasting time if max_limit is far
            # If max_limit is tight, we must search all.
            
            while x <= max_limit:
                new_diffs = {x - a for a in row}
                
                # Check internal conflict
                if new_diffs & row_diffs:
                    x += 1
                    continue
                
                # Check external conflict
                if new_diffs & used_global:
                    x += 1
                    continue
                
                # Valid candidate
                candidates.append(x)
                
                # If we have enough candidates for randomization, stop
                if len(candidates) >= 10:
                    break
                
                x += 1
            
            if not candidates:
                return None
            
            # Choose candidate
            if randomness > 0 and random.random() < randomness:
                # Pick a random candidate from the found ones
                # This can help if the smallest candidate leads to a dead end later?
                # In greedy row construction, there is no "later" in the same row, 
                # but the choice affects the 'used_global' for subsequent rows.
                chosen = random.choice(candidates)
            else:
                chosen = candidates[0]
            
            row.append(chosen)
            row_diffs.update({chosen - a for a in row[:-1]})
            
        return row

    # Main SA Loop
    best_rows = None
    best_scope = float('inf')
    
    # Initial Construction
    used = set()
    rows = []
    for i in range(n_rows):
        # Start with a generous limit, we will shrink it later
        # The theoretical bound is 105, but let's start higher to ensure we get a valid set
        # Then SA will try to minimize.
        # Let's estimate a starting scope. 
        # For n=7, k=5, best is 112. Let's start with 150.
        initial_limit = 200 
        row = build_row_greedy_correct(used, k, initial_limit, randomness=0.2)
        if row is None:
             # Fallback: try with higher limit
             row = build_row_greedy_correct(used, k, 500, randomness=0.0)
             if row is None:
                 # Emergency fallback: simple greedy without limit
                 row = [0]
                 for _ in range(k):
                     x = row[-1] + 1
                     while True:
                         new_diffs = {x - a for a in row}
                         if not (new_diffs & used) and len(new_diffs) == len(row):
                             break
                         x += 1
                     row.append(x)
                     used.update(new_diffs)
        else:
             for i_r in range(len(row)):
                for j_r in range(i_r+1, len(row)):
                    used.add(row[j_r] - row[i_r])
        rows.append(row)
    
    current_rows = [r[:] for r in rows]
    current_scope = get_scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Calculate initial used set for SA
    def calc_used(rows):
        u = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    u.add(r[j] - r[i])
        return u

    current_used = calc_used(current_rows)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    end_time = start_time + time_limit
    
    while time.time() < end_time:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Single element move
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            j = random.randint(1, len(row) - 1)
            
            old_val = row[j]
            
            # Determine range for new value
            lower_bound = row[j-1] + 1
            if j < len(row) - 1:
                upper_bound = row[j+1] - 1
            else:
                # Allow expanding? No, we want to minimize. 
                # But sometimes we need to expand to allow other rows to shrink?
                # In this problem, rows are independent except for diff conflicts.
                # Expanding a row makes it harder to shrink later.
                # Let's restrict to shrinking or same.
                upper_bound = old_val - 1
                
            if lower_bound > upper_bound:
                # Cannot shrink this element. Try a different move or skip.
                continue
                
            new_val = random.randint(lower_bound, upper_bound)
            
            if new_val == old_val:
                continue
            
            # Check validity
            # 1. Internal row validity
            # 2. External conflict
            
            # Temporarily update
            row[j] = new_val
            
            new_diffs_row = get_row_diffs(row)
            if new_diffs_row is None:
                row[j] = old_val
                continue
            
            # Check external conflicts
            # Remove old diffs of this row from current_used
            old_diffs_row = get_row_diffs(row) # Wait, row is modified. Need old diffs.
            # Let's calculate old_diffs before modification.
            
            # Refined Move:
            row[j] = old_val # revert
            
            old_diffs_row = get_row_diffs(row)
            
            row[j] = new_val
            new_diffs_row = get_row_diffs(row)
            
            if new_diffs_row is None:
                row[j] = old_val
                continue
            
            # Check if new_diffs_row conflicts with (current_used - old_diffs_row)
            # i.e., any d in new_diffs_row that is in current_used and NOT in old_diffs_row is a conflict.
            conflict = False
            for d in new_diffs_row:
                if d in current_used:
                    if d not in old_diffs_row:
                        conflict = True
                        break
            
            if conflict:
                row[j] = old_val
                continue
            
            # Acceptance
            new_scope = get_scope(current_rows)
            delta = new_scope - current_scope
            
            if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                # Accept
                current_used -= old_diffs_row
                current_used |= new_diffs_row
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                # Reject
                row[j] = old_val
                
        else:
            # Row Regeneration Move
            # Pick a row, remove it, and try to rebuild it with a tighter limit
            r_idx = random.randint(0, n_rows - 1)
            old_row = current_rows[r_idx]
            old_diffs = get_row_diffs(old_row)
            
            # Remove old row's diffs
            current_used -= old_diffs
            
            # Try to build a new row with a limit based on current best scope
            # We want to see if we can fit a row within best_scope
            # Or even current_scope
            limit_to_try = best_scope
            new_row = build_row_greedy_correct(current_used, k, limit_to_try, randomness=0.3)
            
            if new_row is not None:
                new_diffs = get_row_diffs(new_row)
                # Check if new_diffs conflicts with current_used
                # Since build_row_greedy_correct already checked against current_used, 
                # and current_used does NOT contain old_diffs (we removed them),
                # it should be valid.
                
                # Acceptance
                current_rows[r_idx] = new_row
                current_used |= new_diffs
                new_scope = get_scope(current_rows)
                delta = new_scope - current_scope
                
                if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                else:
                    # Revert
                    current_rows[r_idx] = old_row
                    current_used -= new_diffs
                    current_used |= old_diffs
            else:
                # Failed to build new row, revert (do nothing, old row stays)
                current_used |= old_diffs
            
        # Cool down
        T *= cooling_rate
        
        # If we hit the theoretical lower bound, stop
        if best_scope <= 105:
            break

    if best_rows is None:
        # Fallback
        return current_rows
        
    return best_rows
