import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initial Construction (Greedy Randomized)
    # We want a valid starting point if possible, or at least a close one.
    # Greedy: for each row, add smallest possible x such that no diff conflict.
    def greedy_init():
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

    rows = greedy_init()
    
    # Data structures for fast conflict checking
    # diff_count: dict mapping diff value -> count
    # excess: total excess copies (sum of count[d]-1 for count[d]>1)
    diff_count = {}
    excess = 0
    
    def update_diff_count(diff, delta):
        """Update diff_count and excess for a difference 'diff' changing by 'delta' (+1 or -1)."""
        nonlocal excess
        old_count = diff_count.get(diff, 0)
        new_count = old_count + delta
        if new_count < 0:
            raise ValueError("Negative count")
        
        if old_count == 1 and new_count == 2:
            excess += 1
        elif old_count == 2 and new_count == 1:
            excess -= 1
        elif old_count > 1 and new_count > 1:
            # Excess increases by 1 if count goes up, decreases by 1 if count goes down
            # e.g., 2->3: excess 1->2 (+1). 3->2: excess 2->1 (-1).
            excess += delta
            
        if new_count == 0:
            del diff_count[diff]
        else:
            diff_count[diff] = new_count

    # Initialize diff_count from rows
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                update_diff_count(d, 1)
                
    current_max_entry = max(max(r) for r in rows)
    
    # Cost function: max_entry + PENALTY * excess
    PENALTY = 1000
    
    def get_cost():
        return current_max_entry + PENALTY * excess

    current_cost = get_cost()
    best_rows = [r[:] for r in rows]
    best_cost = current_cost
    best_valid_max = current_max_entry if excess == 0 else float('inf')
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    min_T = 1e-5
    steps = 0
    
    while T > min_T and time.time() < deadline:
        # Pick a random row and a random index to move
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # We can move any index from 1 to k (0 is fixed)
        # To maintain sorted order, we pick an index and restrict the new value
        j_idx = random.randint(1, k)
        
        old_val = row[j_idx]
        
        # Determine valid range for new_val
        # Must be strictly between row[j_idx-1] and row[j_idx+1] (if exists)
        low_bound = row[j_idx - 1] + 1
        high_bound = row[j_idx + 1] - 1 if j_idx + 1 <= k else current_max_entry + 50
        
        # If the range is too small, skip or expand?
        # For a valid row, low_bound < old_val < high_bound.
        # If we want to move, we need space.
        if low_bound >= high_bound:
            # No space to move without breaking order? 
            # Actually, we can swap? No, we are moving a single element.
            # If no integer exists in (low_bound, high_bound) other than old_val, 
            # we can't move to a *different* integer while keeping order.
            # We could try to move to a position that breaks order and then re-sort? 
            # That's complex. Let's stick to moves that preserve local order.
            # If range is empty, we skip this move.
            T *= cooling
            continue
            
        # Choose a new value. 
        # Strategy: Randomly pick from a window around old_val, or uniformly in range.
        # To allow large jumps, let's pick uniformly in [low_bound, high_bound].
        # But we should bias towards values that might reduce cost.
        # For now, uniform random in the valid range.
        # To avoid getting stuck, we can also try values slightly outside if we allow re-sorting?
        # No, let's keep it simple: preserve order.
        
        new_val = random.randint(low_bound, high_bound)
        if new_val == old_val:
            T *= cooling
            continue

        # Evaluate move
        # 1. Remove old diffs
        old_diffs = []
        for i in range(len(row)):
            if i != j_idx:
                d = abs(row[i] - old_val)
                old_diffs.append(d)
                update_diff_count(d, -1)
        
        # 2. Set new val
        row[j_idx] = new_val
        new_max_entry = max(max(r) for r in rows) # This is O(n_rows * k). Can be optimized.
        
        # 3. Add new diffs
        new_diffs = []
        for i in range(len(row)):
            if i != j_idx:
                d = abs(row[i] - new_val)
                new_diffs.append(d)
                update_diff_count(d, 1)
                
        new_cost = new_max_entry + PENALTY * excess
        
        # Acceptance
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / T):
            # Accept
            current_cost = new_cost
            current_max_entry = new_max_entry
            if excess == 0 and new_max_entry < best_valid_max:
                best_valid_max = new_max_entry
                best_rows = [r[:] for r in rows]
                best_cost = current_cost
        else:
            # Revert
            # Remove new diffs
            for d in new_diffs:
                update_diff_count(d, -1)
            # Restore old val
            row[j_idx] = old_val
            # Add old diffs
            for d in old_diffs:
                update_diff_count(d, 1)
            
        T *= cooling
        steps += 1

    # Return the best valid solution found
    # If no valid solution was found (excess never reached 0), we should return the best *valid* one from init or try to fix?
    # The greedy init is always valid. So best_valid_max is at most the init max.
    # We must return a valid solution.
    # If best_valid_max is inf, it means we never found a valid solution during SA, 
    # but the initial one was valid. So we should track the initial one as best_valid.
    
    # Re-initialize best tracking to be safe
    if excess == 0:
        # Current is valid
        pass # Already handled in loop? No, init is valid.
        
    # Check if init was better
    init_max = max(max(r) for r in greedy_init()) # Re-calc? No, we modified rows.
    # We stored best_rows.
    
    return best_rows
