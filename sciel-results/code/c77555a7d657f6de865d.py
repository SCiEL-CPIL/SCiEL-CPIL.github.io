import time
import random
import copy
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    # 145s safety margin for 150s kill limit
    deadline = start_time + 145.0
    
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

    # 1. Greedy Initial Solution
    best_rows = greedy_construct()
    best_scope = max(max(row) for row in best_rows)
    
    # 2. State Initialization
    current_rows = copy.deepcopy(best_rows)
    
    def calculate_initial_state(rows):
        diff_counter = Counter()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff_counter[row[j] - row[i]] += 1
        # Conflicts is the sum of (count - 1) for all differences with count > 1
        conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
        scope = max(max(row) for row in rows)
        return diff_counter, conflicts, scope

    diff_counter, curr_conflicts, curr_scope = calculate_initial_state(current_rows)
    
    def get_scope(rows):
        return max(max(row) for row in rows)

    # Helper to update diff_counter and curr_conflicts incrementally
    # Returns the change in conflicts
    def apply_move(r, j, old_val, new_val, diff_counter, curr_conflicts):
        row = current_rows[r]
        
        # 1. Remove old differences
        # Differences involving index j: (j, i) for i<j and (i, j) for i>j
        # Note: In our row representation [a0, a1, ...], diffs are a_j - a_i for i < j
        # So for index j, we have diffs: a_j - a_0, ..., a_j - a_{j-1}
        # And diffs where j is the lower index: a_{j+1} - a_j, ..., a_{k} - a_j
        
        for i in range(j):
            d = old_val - row[i]
            c = diff_counter[d]
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
            # If count went from 1 to 0, conflicts decreased by 0 (was 0 excess)
            # If count went from 2 to 1, conflicts decreased by 1 (was 1 excess)
            # If count went from c to c-1, excess goes from c-1 to c-2. Delta = -1 if c > 1 else 0.
            if c > 1:
                curr_conflicts -= 1

        for i in range(j+1, len(row)):
            d = row[i] - old_val
            c = diff_counter[d]
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
            if c > 1:
                curr_conflicts -= 1

        # 2. Apply new value
        current_rows[r][j] = new_val

        # 3. Add new differences
        for i in range(j):
            d = new_val - row[i]
            c = diff_counter.get(d, 0)
            diff_counter[d] = c + 1
            # If count went from 0 to 1, excess goes from 0 to 0. Delta = 0.
            # If count went from 1 to 2, excess goes from 0 to 1. Delta = +1.
            # If count went from c to c+1, excess goes from c-1 to c. Delta = +1 if c >= 1 else 0.
            if c >= 1:
                curr_conflicts += 1

        for i in range(j+1, len(row)):
            d = row[i] - new_val
            c = diff_counter.get(d, 0)
            diff_counter[d] = c + 1
            if c >= 1:
                curr_conflicts += 1

        return curr_conflicts

    def revert_move(r, j, old_val, new_val, diff_counter, curr_conflicts):
        # Simply apply the move again in reverse: (new_val -> old_val)
        return apply_move(r, j, new_val, old_val, diff_counter, curr_conflicts)

    T = 50.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Phase 1: Targeted Local Search (Hill Climbing)
        # Try to reduce the scope by moving elements in the row with the max value down.
        improved = False
        
        # Find the row with the maximum value
        max_row_idx = -1
        max_val = -1
        for r in range(n_rows):
            if current_rows[r][-1] > max_val:
                max_val = current_rows[r][-1]
                max_row_idx = r
        
        # Try to reduce elements in max_row_idx, starting from the end (largest values)
        # We iterate positions from k down to 1
        for j in range(k, 0, -1):
            if time.time() > deadline:
                break
            row = current_rows[max_row_idx]
            old_val = row[j]
            
            # Try to reduce old_val by 1, 2, ... until we hit a valid configuration or the previous element
            # We only care about reducing the scope, so we only look for valid moves (conflicts == 0)
            # that result in a lower scope.
            
            # Optimization: If j < k, the max of the row might not change if row[j] is not the max.
            # But usually the last element is the max. Let's just try to reduce any element in the max row.
            
            min_val = row[j-1] + 1
            # We only want to reduce, so new_val < old_val
            # We stop if new_val would not reduce the scope of the row (if this element was the max)
            # Or if it reduces the global scope.
            
            # To be efficient, we can just try old_val - 1, old_val - 2...
            # But scanning down 1 by 1 might be slow if the gap is large.
            # However, usually gaps are small.
            
            # Let's try a random jump down first? No, hill climbing is better for polish.
            # Let's try reducing by 1. If it fails, try 2? 
            # Actually, let's just try to find ANY valid value lower than old_val.
            # We can scan downwards.
            
            # To avoid O(N) scan every time, we might just try a few steps.
            # But since we want to minimize scope, we want the *lowest* valid value?
            # No, we just want *any* valid value that reduces the max.
            # If we reduce row[j] to v, and v < old_val, and v is valid, 
            # does it reduce the global scope?
            # Only if old_val was the unique global max.
            
            # Let's check if old_val is the global max.
            # If yes, reducing it reduces scope.
            # If not, reducing it might allow other rows to be reduced later, but doesn't reduce scope directly.
            # So we focus on the element that IS the global max.
            
            if old_val != max_val:
                continue
                
            # Scan downwards
            found = False
            # Limit the scan to a reasonable number of steps to avoid getting stuck in a long scan
            # If we can't find a valid value within 100 steps, we give up on this position for now.
            for delta in range(1, 100):
                new_val = old_val - delta
                if new_val < min_val:
                    break
                
                # Check if this new_val is valid
                # We need to simulate the move
                # To be fast, we can check conflicts incrementally without committing?
                # Or commit, check, revert.
                
                # Let's commit, check, revert.
                new_conflicts = apply_move(max_row_idx, j, old_val, new_val, diff_counter, curr_conflicts)
                
                if new_conflicts == 0:
                    # Valid move!
                    # Does it reduce scope?
                    # Since old_val was the max, and new_val < old_val, 
                    # the new max of this row is max(row[j-1], new_val, ... row[k])
                    # If new_val is still the max of the row, the global max becomes new_val (if it was unique)
                    # or remains the same if another row has the same max.
                    
                    new_scope = get_scope(current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = copy.deepcopy(current_rows)
                        curr_conflicts = new_conflicts
                        curr_scope = new_scope
                        improved = True
                        found = True
                        break
                    else:
                        # It's a valid move but didn't reduce scope. 
                        # We can accept it to escape local minima? 
                        # For now, let's only accept if it improves scope to keep it simple and safe.
                        # Revert
                        curr_conflicts = revert_move(max_row_idx, j, old_val, new_val, diff_counter, new_conflicts)
                else:
                    # Invalid, revert
                    curr_conflicts = revert_move(max_row_idx, j, old_val, new_val, diff_counter, new_conflicts)
            
            if found:
                break # Restart the row search with the new state
            
        if improved:
            continue # Loop back to try more improvements
            
        # Phase 2: SA Perturbation (Kick)
        # If we are stuck, perturb the state.
        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k)
        
        old_val = current_rows[r][j]
        
        min_val = current_rows[r][j-1] + 1
        max_val = max(old_val + 10, best_scope)
        if j < k:
            max_val = min(max_val, current_rows[r][j+1] - 1)
            
        if min_val > max_val:
            T *= cooling
            continue
            
        new_val = random.randint(min_val, max_val)
        if new_val == old_val:
            T *= cooling
            continue
            
        # Apply move
        new_conflicts = apply_move(r, j, old_val, new_val, diff_counter, curr_conflicts)
        
        new_scope = get_scope(current_rows)
        
        # Energy function: Scope + Penalty for conflicts
        # We want to minimize scope, but we must maintain validity (conflicts=0) eventually.
        # For SA, we allow conflicts.
        delta_scope = new_scope - curr_scope
        delta_conflicts = new_conflicts - curr_conflicts
        delta_energy = delta_scope + 10 * delta_conflicts
        
        accept = False
        if delta_energy < 0:
            accept = True
        else:
            if T > 0:
                # Avoid overflow in exp
                if delta_energy / T < 700:
                    accept = random.random() < math.exp(-delta_energy / T)
        
        if accept:
            curr_scope = new_scope
            curr_conflicts = new_conflicts
            if new_conflicts == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = copy.deepcopy(current_rows)
        else:
            # Revert
            curr_conflicts = revert_move(r, j, old_val, new_val, diff_counter, new_conflicts)
            
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_rows
