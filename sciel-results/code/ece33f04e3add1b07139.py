import time
import random
import copy
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Helper to calculate conflicts and scope efficiently
    def calculate_state(rows):
        diff_counter = Counter()
        for row in rows:
            for i in range(len(row)):
                ai = row[i]
                for j in range(i+1, len(row)):
                    diff_counter[row[j] - ai] += 1
        conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
        scope = max(max(row) for row in rows)
        return diff_counter, conflicts, scope

    # Helper to get scope
    def get_scope(rows):
        return max(max(row) for row in rows)

    # Initialization Strategies
    
    def greedy_sequential():
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

    def greedy_balanced():
        """
        Greedy that tries to balance the growth of rows.
        Instead of finishing one row completely before starting the next,
        it extends the row with the smallest current maximum value.
        """
        used = set()
        rows = [[0] for _ in range(n_rows)]
        next_candidate = [0] * n_rows # next x to try for each row
        
        while any(len(row) < k + 1 for row in rows):
            # Find row with smallest max that is not complete
            min_max = float('inf')
            target_row_idx = -1
            for r in range(n_rows):
                if len(rows[r]) < k + 1:
                    current_max = rows[r][-1]
                    if current_max < min_max:
                        min_max = current_max
                        target_row_idx = r
            
            if target_row_idx == -1:
                break
                
            r = target_row_idx
            row = rows[r]
            
            # Find next valid x for this row
            x = next_candidate[r]
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    next_candidate[r] = x
                    break
        
        return rows

    def greedy_randomized(order_seed):
        """
        Greedy with randomized row ordering.
        """
        random.seed(order_seed)
        used = set()
        rows = []
        row_indices = list(range(n_rows))
        random.shuffle(row_indices)
        
        for r_idx in row_indices:
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        # Reorder back to original indices? Not strictly necessary for validity, 
        # but keeps consistency. The set of rows is what matters.
        # However, for local search, fixed indices are easier.
        # We'll just return the list of rows as generated.
        return rows

    # Generate initial candidates
    candidates = []
    
    # 1. Standard Greedy
    candidates.append(greedy_sequential())
    
    # 2. Balanced Greedy
    candidates.append(greedy_balanced())
    
    # 3. A few randomized greedy variants
    for seed in [1, 42, 123, 999]:
        candidates.append(greedy_randomized(seed))
        
    # Select the best initial candidate based on scope
    best_init = min(candidates, key=get_scope)
    best_scope = get_scope(best_init)
    best_rows = copy.deepcopy(best_init)
    
    current_rows = copy.deepcopy(best_init)
    diff_counter, curr_conflicts, curr_scope = calculate_state(current_rows)
    
    # If the best init is already valid and has a very low scope, we might be done,
    # but we should still try to shrink further.
    
    def try_shrink_all():
        """
        Attempt to shrink every element by 1 repeatedly until no improvement.
        This is a deterministic local search step.
        """
        nonlocal best_scope, best_rows, curr_scope, curr_conflicts
        progress = True
        # Limit iterations to avoid infinite loops if logic is flawed, 
        # though logically it should terminate as scope decreases.
        max_passes = 100
        passes = 0
        
        while progress and time.time() < deadline and passes < max_passes:
            progress = False
            passes += 1
            # Iterate rows in random order to avoid bias
            row_order = list(range(n_rows))
            random.shuffle(row_order)
            
            for r in row_order:
                # Iterate elements from largest to smallest index
                for j in range(k, 0, -1):
                    old_val = current_rows[r][j]
                    new_val = old_val - 1
                    
                    if new_val <= current_rows[r][j-1]:
                        continue
                    
                    # Remove old differences
                    old_diffs = []
                    for i in range(j):
                        d = old_val - current_rows[r][i]
                        old_diffs.append(d)
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]
                    
                    for i in range(j+1, len(current_rows[r])):
                        d = current_rows[r][i] - old_val
                        old_diffs.append(d)
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]
                    
                    # Apply new value
                    current_rows[r][j] = new_val
                    
                    # Add new differences
                    for i in range(j):
                        d = new_val - current_rows[r][i]
                        diff_counter[d] += 1
                    
                    for i in range(j+1, len(current_rows[r])):
                        d = current_rows[r][i] - new_val
                        diff_counter[d] += 1
                    
                    new_conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
                    
                    if new_conflicts == 0:
                        new_scope = max(max(row) for row in current_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = copy.deepcopy(current_rows)
                        curr_scope = new_scope
                        curr_conflicts = new_conflicts
                        progress = True
                    else:
                        # Revert
                        current_rows[r][j] = old_val
                        for d in old_diffs:
                            diff_counter[d] += 1
                        for i in range(j):
                            d = new_val - current_rows[r][i]
                            diff_counter[d] -= 1
                            if diff_counter[d] == 0:
                                del diff_counter[d]
                        for i in range(j+1, len(current_rows[r])):
                            d = current_rows[r][i] - new_val
                            diff_counter[d] -= 1
                            if diff_counter[d] == 0:
                                del diff_counter[d]

    # Initial shrink pass
    try_shrink_all()
    
    # Simulated Annealing Loop
    T = 50.0
    cooling = 0.995
    no_improve_count = 0
    restart_threshold = 2000
    max_restarts = 10
    restarts = 0
    kick_threshold = 500
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try a few shrink steps
        # We already did a full pass, but maybe some new opportunities arose after SA?
        # Actually, SA moves might open up shrink opportunities.
        # Let's do a quick random shrink attempt instead of full pass for speed.
        for _ in range(10):
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            
            old_val = current_rows[r][j]
            new_val = old_val - 1
            
            if new_val <= current_rows[r][j-1]:
                continue
            
            old_diffs = []
            for i in range(j):
                d = old_val - current_rows[r][i]
                old_diffs.append(d)
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            for i in range(j+1, len(current_rows[r])):
                d = current_rows[r][i] - old_val
                old_diffs.append(d)
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            current_rows[r][j] = new_val
            
            for i in range(j):
                d = new_val - current_rows[r][i]
                diff_counter[d] += 1
            
            for i in range(j+1, len(current_rows[r])):
                d = current_rows[r][i] - new_val
                diff_counter[d] += 1
            
            new_conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
            
            if new_conflicts == 0:
                new_scope = max(max(row) for row in current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = copy.deepcopy(current_rows)
                    improved = True
                    no_improve_count = 0
                curr_scope = new_scope
                curr_conflicts = new_conflicts
                break
            else:
                # Revert
                current_rows[r][j] = old_val
                for d in old_diffs:
                    diff_counter[d] += 1
                for i in range(j):
                    d = new_val - current_rows[r][i]
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for i in range(j+1, len(current_rows[r])):
                    d = current_rows[r][i] - new_val
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
        
        if improved:
            # Reset temperature slightly or keep it
            T = max(T, 10.0)
            continue
        
        # Phase 2: SA Move
        # Decide if we need a "kick" (larger perturbation)
        if no_improve_count > kick_threshold:
            # Kick: move 2 random elements
            for _ in range(2):
                r = random.randint(0, n_rows - 1)
                j = random.randint(1, k)
                old_val = current_rows[r][j]
                min_val = current_rows[r][j-1] + 1
                max_val = max(old_val + 10, best_scope)
                if j < k:
                    max_val = min(max_val, current_rows[r][j+1] - 1)
                
                if min_val > max_val:
                    continue
                
                new_val = random.randint(min_val, max_val)
                if new_val == old_val:
                    continue
                
                # Remove old
                old_diffs = []
                for i in range(j):
                    d = old_val - current_rows[r][i]
                    old_diffs.append(d)
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for i in range(j+1, len(current_rows[r])):
                    d = current_rows[r][i] - old_val
                    old_diffs.append(d)
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                
                # Apply
                current_rows[r][j] = new_val
                
                # Add new
                for i in range(j):
                    d = new_val - current_rows[r][i]
                    diff_counter[d] += 1
                for i in range(j+1, len(current_rows[r])):
                    d = current_rows[r][i] - new_val
                    diff_counter[d] += 1
                    
                # Do not check validity or accept/reject here, just force it to escape
                # Update state
                curr_scope = max(max(row) for row in current_rows)
                curr_conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
            
            no_improve_count = 0
            T = 50.0 # Reset temp
            continue

        # Standard SA move
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k)
        
        old_val = current_rows[r][j]
        
        min_val = current_rows[r][j-1] + 1
        max_val = max(old_val + 5, best_scope)
        if j < k:
            max_val = min(max_val, current_rows[r][j+1] - 1)
            
        if min_val > max_val:
            no_improve_count += 1
            continue
            
        # Bias towards smaller values
        if random.random() < 0.7:
            new_val = random.randint(min_val, min(old_val, max_val))
        else:
            new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            no_improve_count += 1
            continue
            
        # Remove old diffs
        old_diffs = []
        for i in range(j):
            d = old_val - current_rows[r][i]
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        for i in range(j+1, len(current_rows[r])):
            d = current_rows[r][i] - old_val
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        
        # Apply new
        current_rows[r][j] = new_val
        
        # Add new diffs
        for i in range(j):
            d = new_val - current_rows[r][i]
            diff_counter[d] += 1
        for i in range(j+1, len(current_rows[r])):
            d = current_rows[r][i] - new_val
            diff_counter[d] += 1
            
        new_scope = max(max(row) for row in current_rows)
        new_conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
        
        delta_scope = new_scope - curr_scope
        delta_conflicts = new_conflicts - curr_conflicts
        delta_energy = delta_scope + 20 * delta_conflicts
        
        accept = False
        if delta_energy < 0:
            accept = True
        else:
            if T > 0:
                accept = random.random() < math.exp(-delta_energy / T)
        
        if accept:
            curr_scope = new_scope
            curr_conflicts = new_conflicts
            if new_conflicts == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = copy.deepcopy(current_rows)
                improved = True
                no_improve_count = 0
        else:
            # Revert
            current_rows[r][j] = old_val
            for d in old_diffs:
                diff_counter[d] += 1
            for i in range(j):
                d = new_val - current_rows[r][i]
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for i in range(j+1, len(current_rows[r])):
                d = current_rows[r][i] - new_val
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
        
        if not improved:
            no_improve_count += 1
        
        # Cool down
        T *= cooling
        if T < 0.1:
            T = 0.1

        # Restart if stuck
        if no_improve_count > restart_threshold:
            restarts += 1
            if restarts >= max_restarts:
                break
            
            # Restart with a new random greedy init
            new_init = greedy_randomized(restarts)
            new_scope = get_scope(new_init)
            if new_scope < best_scope or restarts % 2 == 0: # Always restart if worse, or alternate
                current_rows = copy.deepcopy(new_init)
                diff_counter, curr_conflicts, curr_scope = calculate_state(current_rows)
                try_shrink_all()
            
            no_improve_count = 0
            T = 50.0

    return best_rows
