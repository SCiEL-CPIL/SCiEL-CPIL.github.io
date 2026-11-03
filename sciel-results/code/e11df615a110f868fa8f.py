import time
import random
import numpy as np

def construct(n_rows, k):
    # Helper to calculate differences and check validity
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Greedy to get initial bound
    def greedy_construct():
        used, rows = set(), []
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

    start_time = time.time()
    time_limit = 110.0
    
    # 1. Get a valid solution with Greedy
    rows_greedy = greedy_construct()
    best_rows = [list(r) for r in rows_greedy]
    best_max_val = max(max(r) for r in best_rows)
    
    # 2. Simulated Annealing to improve (reduce max value)
    # State: list of rows, each row is list of k+1 ints (sorted)
    # We represent state as list of lists.
    # Move: Pick a random row, a random index (1..k, since 0 is fixed), 
    # and replace the value with a new random value in [1, current_limit].
    # Then sort the row.
    
    # We need to track global difference counts to update fast.
    # diff_count[d] = number of rows containing difference d
    
    def initialize_state(limit):
        state = []
        diff_count = {}
        for _ in range(n_rows):
            # Generate k random distinct numbers in [1, limit]
            row_vals = [0] + random.sample(range(1, limit + 1), k)
            row_vals.sort()
            state.append(row_vals)
            
            # Update diff counts
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row_vals[j] - row_vals[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
        return state, diff_count

    def calc_energy(state, diff_count):
        # Energy = number of conflicts (d with count > 1) * 1000 + max_val
        # Actually, we want to minimize max_val primarily?
        # No, validity is hard constraint. 
        # Let's define Energy = num_conflicts * 10000 + max_val
        # If num_conflicts == 0, we minimize max_val.
        max_val = max(max(r) for r in state)
        conflicts = 0
        for d, c in diff_count.items():
            if c > 1:
                conflicts += c - 1
        return conflicts * 10000 + max_val

    def make_move(state, diff_count, limit):
        # Returns new_state, new_diff_count, delta_energy
        # Pick random row
        r_idx = random.randint(0, n_rows - 1)
        row = state[r_idx]
        
        # Pick random position to change (1 to k, 0 is fixed)
        pos = random.randint(1, k)
        old_val = row[pos]
        
        # Pick new value
        new_val = random.randint(1, limit)
        
        if new_val == old_val:
            return state, diff_count, 0, False # No change
            
        # Check if new_val already in row (excluding old_val)
        # Since row is sorted, we can just check set
        if new_val in row:
             # If new_val is in row, it's a duplicate in the row, which is invalid.
             # We can treat this as a high penalty move or just reject.
             # Let's just reject to keep state valid w.r.t row uniqueness.
             return state, diff_count, 0, False

        # Update diff_count for the old row
        for i in range(k + 1):
            if i == pos: continue
            d_old = abs(row[i] - old_val)
            diff_count[d_old] -= 1
            if diff_count[d_old] == 0:
                del diff_count[d_old]
                
        # Create new row
        new_row = list(row)
        new_row[pos] = new_val
        new_row.sort()
        
        # Update diff_count for the new row
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d_new = new_row[j] - new_row[i]
                diff_count[d_new] = diff_count.get(d_new, 0) + 1
                
        # Note: The loop above adds ALL diffs. We removed the old ones involving old_val.
        # But the diffs between other elements in the row are unchanged.
        # Wait, the logic above:
        # 1. Remove diffs involving old_val.
        # 2. Add ALL diffs of new_row.
        # This is correct because the diffs not involving the changed element are re-added with the same value,
        # so their count goes -1 (from remove? No, we didn't remove them) ... 
        # Ah, my remove step only removed diffs involving `old_val`.
        # The diffs NOT involving `old_val` (e.g., row[0]-row[1] if pos is 2) are still in diff_count.
        # When I add ALL diffs of new_row, I add those unchanged diffs again.
        # So I need to be careful.
        
        # Correct Update Logic:
        # 1. Remove all diffs of the OLD row from diff_count.
        # 2. Add all diffs of the NEW row to diff_count.
        
        # Let's restart the move logic for clarity and correctness.
        
        return state, diff_count, 0, False # Placeholder, will implement proper logic below

    # Let's implement a cleaner move function
    def apply_move(state, diff_count, limit):
        r_idx = random.randint(0, n_rows - 1)
        row = state[r_idx]
        pos = random.randint(1, k)
        old_val = row[pos]
        new_val = random.randint(1, limit)
        
        if new_val == old_val:
            return state, diff_count, 0, False

        if new_val in row:
            return state, diff_count, 0, False

        # 1. Remove old row diffs
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = row[j] - row[i]
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
        
        # 2. Create new row
        new_row = list(row)
        new_row[pos] = new_val
        new_row.sort()
        
        # 3. Add new row diffs
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = new_row[j] - new_row[i]
                diff_count[d] = diff_count.get(d, 0) + 1
        
        state[r_idx] = new_row
        return state, diff_count, 0, True

    # SA Main Loop
    # We will try to find a valid solution (conflicts=0) with min max_val.
    # Start with limit = best_max_val
    # If we find a valid one, we can try to reduce limit.
    
    # To save time, we'll run SA for a fixed number of iterations or time.
    # We'll try to improve upon the greedy result.
    
    current_limit = best_max_val
    state, diff_count = initialize_state(current_limit)
    
    # Calculate initial energy
    def get_energy(state, diff_count):
        max_val = max(max(r) for r in state)
        conflicts = 0
        for c in diff_count.values():
            if c > 1:
                conflicts += c - 1
        return conflicts * 10000 + max_val

    curr_energy = get_energy(state, diff_count)
    best_energy = curr_energy
    best_state = [list(r) for r in state]
    
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit - 5:
        if T < T_min:
            break
            
        new_state, new_diff_count, _, moved = apply_move(state, diff_count, current_limit)
        if not moved:
            continue
            
        new_energy = get_energy(new_state, new_diff_count)
        delta = new_energy - curr_energy
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            state = new_state
            diff_count = new_diff_count
            curr_energy = new_energy
            
            if curr_energy < best_energy:
                best_energy = curr_energy
                best_state = [list(r) for r in state]
                
                # If we found a valid solution (conflicts=0), 
                # we can try to reduce the limit to force smaller values.
                # Check if max_val is less than current_limit
                curr_max = max(max(r) for r in state)
                if curr_max < current_limit and curr_energy < 10000: # Valid
                    # We can lower the limit to the current max to prevent expanding
                    # Or just note that we found a better max.
                    # Let's update best_max_val globally if this is better
                    if curr_max < best_max_val:
                        best_max_val = curr_max
                        # Optionally reduce current_limit to tighten search
                        current_limit = curr_max
                        # Re-initialize? No, just keep going. The moves are constrained by current_limit.
                        # If we lower current_limit, we might get stuck if current state has values > current_limit?
                        # No, we set current_limit = curr_max, so all values are <= current_limit.
                        # So it's safe.
                        
        T *= cooling_rate
        
    # Check if best_state is valid
    # If yes, return it. If no, return greedy.
    # We need to ensure best_state is valid before returning.
    # Energy < 10000 implies 0 conflicts.
    if best_energy < 10000:
        # Verify validity just in case
        all_diffs = set()
        valid = True
        for r in best_state:
            r_diffs = get_diffs(r)
            if r_diffs & all_diffs:
                valid = False
                break
            all_diffs |= r_diffs
        if valid:
            return best_state
    return best_rows
