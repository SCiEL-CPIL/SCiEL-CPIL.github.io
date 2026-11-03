import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    best_rows = None
    best_scope = float('inf')
    
    # Helper to calculate max entry
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(r) for r in rows)

    # Helper to check validity of a set of rows (for debugging/safety, though we maintain invariants)
    def is_valid(rows):
        used = set()
        for r in rows:
            diffs = set()
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used or d in diffs:
                        return False
                    diffs.add(d)
            used.update(diffs)
        return True

    # Randomized Backtracking Row Builder
    def build_row(used, k, row, candidates_limit=10):
        if len(row) == k + 1:
            return row
        
        last = row[-1]
        # We need to find x > last such that {x - a for a in row} is disjoint from used
        # We only look at the smallest few candidates to keep it fast
        candidates = []
        x = last + 1
        # Limit search space for candidates to avoid infinite loops in sparse areas
        # Though for small k, valid x's are usually close
        while len(candidates) < candidates_limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                candidates.append(x)
                if len(candidates) >= candidates_limit:
                    break
            x += 1
            # Safety break if we go too far (shouldn't happen often for small k)
            if x > 500: 
                break
        
        if not candidates:
            return None
        
        # Randomize the order of candidates to explore different branches
        random.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in row}
            new_used = used | new_diffs
            res = build_row(new_used, k, row + [x], candidates_limit)
            if res is not None:
                return res
        return None

    # Local Search Optimizer
    def optimize(rows, time_budget):
        # rows is a list of lists
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used.add(r[j] - r[i])
        
        current_scope = get_scope(rows)
        best_local_rows = [r[:] for r in rows]
        best_local_scope = current_scope
        
        end_time = time.time() + time_budget
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < end_time and no_improve_count < max_no_improve:
            # Pick a random row and a random position to modify
            r_idx = random.randint(0, len(rows) - 1)
            row = rows[r_idx]
            # Pick a position j > 0
            j = random.randint(1, len(row) - 1)
            
            old_val = row[j]
            # Try to decrease the value
            # New value must be > row[j-1] and < row[j] (if j < len(row)-1)
            # Actually, if we change row[j], it affects diffs (j-1, j), (j, j+1), etc.
            # It's safer to try a random value in the range (row[j-1], old_val)
            # If j is the last element, range is (row[j-1], old_val)
            # If j is not last, we also need new_val < row[j+1]
            
            lower_bound = row[j-1] + 1
            if j < len(row) - 1:
                upper_bound = row[j+1] - 1
            else:
                upper_bound = old_val - 1
                
            if lower_bound > upper_bound:
                no_improve_count += 1
                continue
                
            # Try a few random values in this range
            # Or just try the smallest possible? Random is better for escaping.
            # Let's try a random value
            new_val = random.randint(lower_bound, upper_bound)
            
            if new_val == old_val:
                continue
            
            # Calculate old diffs for this row
            old_diffs = set()
            for i in range(len(row)):
                for l in range(i+1, len(row)):
                    old_diffs.add(row[l] - row[i])
            
            # Calculate new diffs for this row
            # Temporarily update row
            row[j] = new_val
            new_diffs = set()
            for i in range(len(row)):
                for l in range(i+1, len(row)):
                    new_diffs.add(row[l] - row[i])
            
            # Check validity
            # The new diffs must not intersect with (used - old_diffs)
            # i.e., new_diffs & (used - old_diffs) must be empty
            # And new_diffs must be unique within itself (guaranteed by construction if row is strictly increasing? No, diffs can repeat within a row? 
            # Wait, the problem says "all differences ... are distinct over all rows".
            # It does NOT say differences within a single row must be distinct.
            # "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
            # This implies the set of ALL differences from ALL rows must be distinct.
            # So yes, no duplicate differences anywhere.
            
            # So we need:
            # 1. new_diffs is a set (no internal duplicates). If there are internal duplicates, it's invalid.
            if len(new_diffs) != len(list(new_diffs)): # This check is redundant if we use a set, but we need to ensure the *count* matches the number of pairs?
                 # Actually, if a row has duplicate diffs, e.g., [0, 1, 2], diffs are 1, 2, 1. Set is {1, 2}.
                 # The problem requires ALL differences to be distinct. So [0, 1, 2] is INVALID because diff 1 appears twice.
                 # So we must ensure the row itself is a Golomb Ruler (distinct diffs within the row).
                 pass
            
            # Correct check:
            # 1. The new row must have all distinct differences internally.
            # 2. The new differences must not conflict with differences from other rows.
            
            # Let's re-verify internal distinctness
            # If new_diffs size < (k+1)*k/2, then there are duplicates.
            num_pairs = (len(row) * (len(row) - 1)) // 2
            if len(new_diffs) < num_pairs:
                row[j] = old_val
                no_improve_count += 1
                continue

            # Check conflict with global used set
            # Global used set currently contains old_diffs.
            # We need to check if new_diffs intersects with (used - old_diffs)
            # Or simpler: 
            # Remove old_diffs from used, check if new_diffs intersects used, if not, add new_diffs back.
            
            # Efficient check:
            # conflict = False
            # for d in new_diffs:
            #     if d in used and d not in old_diffs:
            #         conflict = True
            #         break
            # if d in old_diffs, it's "allowed" to be reused? No, the set of ALL diffs must be distinct.
            # So if d is in old_diffs, it was already in `used`.
            # If we replace old_diffs with new_diffs, we must ensure that any d in new_diffs that is NOT in old_diffs
            # does not exist in `used` (which still contains old_diffs at this moment).
            # Wait, if d is in new_diffs and d is in old_diffs, that's fine, it's just staying in the set.
            # But if d is in new_diffs and d is in used but d NOT in old_diffs, that's a conflict with another row.
            
            conflict = False
            for d in new_diffs:
                if d in used:
                    if d not in old_diffs:
                        conflict = True
                        break
            
            if conflict:
                row[j] = old_val
                no_improve_count += 1
                continue
            
            # If no conflict, accept the move
            used -= old_diffs
            used |= new_diffs
            no_improve_count = 0
            
            current_scope = get_scope(rows)
            if current_scope < best_local_scope:
                best_local_scope = current_scope
                best_local_rows = [r[:] for r in rows]
            else:
                # Optional: Accept non-improving moves with some probability (Simulated Annealing)
                # For simplicity, let's just do Hill Climbing (only accept if better or equal scope? No, only if better scope or just keep exploring?)
                # Standard Hill Climbing: only accept if better. But this can get stuck.
                # Let's accept if scope is not worse? No, that allows drifting.
                # Let's accept if scope is better, OR with a small probability if scope is same (to move to a different local min).
                # If scope is worse, reject.
                if current_scope > best_local_scope:
                    # Reject and revert
                    used -= new_diffs
                    used |= old_diffs
                    row[j] = old_val
                    no_improve_count += 1
                else:
                    # Keep the move (scope is same or better, but we already updated best if better)
                    # If scope is same, we stay here.
                    pass
                    
        return best_local_rows, best_local_scope

    # Main Loop
    attempts = 0
    while time.time() < start_time + time_limit:
        attempts += 1
        # 1. Construct
        used = set()
        rows = []
        success = True
        for i in range(n_rows):
            # Randomize the candidate limit or shuffle seed implicitly by random module state
            row = build_row(used, k, [0], candidates_limit=5 + random.randint(0, 5))
            if row is None:
                success = False
                break
            rows.append(row)
            # Update used
            for j in range(len(row)):
                for l in range(j+1, len(row)):
                    used.add(row[l] - row[j])
        
        if not success:
            continue
            
        # 2. Optimize
        # Allocate time based on remaining
        remaining = start_time + time_limit - time.time()
        if remaining < 5:
            break
            
        opt_time = min(10, remaining / 2) # Try optimizing for up to 10s per construction
        optimized_rows, opt_scope = optimize(rows, opt_time)
        
        if opt_scope < best_scope:
            best_scope = opt_scope
            best_rows = optimized_rows
            
        # If we found the theoretical lower bound (105 for 7,5), we can stop
        if best_scope <= 105:
            break

    if best_rows is None:
        # Fallback to simple greedy if randomization failed completely
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
        best_rows = rows
        
    return best_rows
