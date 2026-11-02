import random
import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Target: n_rows rows, each of length k+1
    # We want to minimize the max entry.
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_solution(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            # Check increasing
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
            diffs = get_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True

    def objective(rows):
        if not is_valid_solution(rows):
            return float('inf')
        return max(max(row) for row in rows)

    def greedy_construct(used_diffs, randomize=False, bias_factor=0.0):
        """
        Construct a single row given the set of already used differences.
        If randomize is True, it might skip some valid small numbers to explore different structures.
        """
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            best_next = None
            candidates = []
            # Search range: from current_max + 1 up to some limit
            # A good heuristic limit is current_max + k + 10 or similar, but we might need to go further.
            # Let's search up to current_max + 50 initially, expanding if needed.
            search_limit = current_max + 100 
            
            for x in range(current_max + 1, search_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    candidates.append(x)
                    if not randomize:
                        break # Greedy: take the first one
                    else:
                        # Collect a few candidates if randomizing
                        if len(candidates) >= 5:
                            break
            
            if not candidates:
                # If we didn't find any in the initial window, expand search
                # This shouldn't happen often if used_diffs isn't too dense, but just in case
                found = False
                for x in range(current_max + 1, current_max + 1000 + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                        found = True
                        if len(candidates) >= 5 and randomize:
                            break
                        if not randomize:
                            break
                if not found:
                    return None # Failed to construct

            if randomize:
                # Choose randomly from candidates, maybe weighted towards smaller ones
                # Bias: prefer smaller numbers
                weights = [1.0 / (c - current_max + 1) for c in candidates]
                chosen = random.choices(candidates, weights=weights)[0]
            else:
                chosen = candidates[0]
            
            row.append(chosen)
            current_max = chosen
            
        # Update used_diffs is done by caller
        return row

    def rebuild_row(rows, row_idx, used_diffs_excl_row, randomize=True):
        """
        Rebuild a specific row, assuming other rows are fixed.
        used_diffs_excl_row is the set of diffs from all OTHER rows.
        """
        # Remove the old row's diffs from the global used set logic (handled by passing excl set)
        new_row = greedy_construct(used_diffs_excl_row, randomize=randomize)
        if new_row is None:
            return None
        return new_row

    # 1. Initial Population
    best_rows = None
    best_obj = float('inf')
    
    # Generate a few initial solutions using greedy with different random seeds
    for i in range(5):
        random.seed(42 + i)
        rows = []
        used = set()
        success = True
        for r in range(n_rows):
            row = greedy_construct(used, randomize=(i > 0), bias_factor=0)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= get_diffs(row)
        
        if success:
            obj = objective(rows)
            if obj < best_obj:
                best_obj = obj
                best_rows = [r[:] for r in rows]
                
    if best_rows is None:
        # Fallback: pure deterministic greedy
        rows = []
        used = set()
        for r in range(n_rows):
            row = greedy_construct(used, randomize=False)
            rows.append(row)
            used |= get_diffs(row)
        best_rows = rows
        best_obj = objective(rows)

    # 2. Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_obj = best_obj
    
    T = 50.0 # Initial temperature
    cooling = 0.999
    
    # Pre-compute diffs for each row for faster updates
    current_diffs = [get_diffs(row) for row in current_rows]
    all_used = set()
    for d in current_diffs:
        all_used |= d

    while time.time() < deadline:
        # Select a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Local Move: Shift one element in one row
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k) # Don't move 0
            delta = random.randint(-5, 5)
            if delta == 0: continue
            
            new_val = current_rows[r_idx][pos] + delta
            # Check bounds
            if new_val <= current_rows[r_idx][pos-1] or (pos < k and new_val >= current_rows[r_idx][pos+1]):
                continue
            
            # Check if new value creates conflicts with other elements in the same row
            # and with other rows
            old_val = current_rows[r_idx][pos]
            
            # Diffs affected in this row:
            # Old diffs involving pos: {old_val - a_i for i < pos} U {a_j - old_val for j > pos}
            # New diffs involving pos: {new_val - a_i for i < pos} U {a_j - new_val for j > pos}
            
            row = current_rows[r_idx]
            old_diffs_pos = set()
            for i in range(pos):
                old_diffs_pos.add(old_val - row[i])
            for j in range(pos+1, k+1):
                old_diffs_pos.add(row[j] - old_val)
                
            new_diffs_pos = set()
            for i in range(pos):
                new_diffs_pos.add(new_val - row[i])
            for j in range(pos+1, k+1):
                new_diffs_pos.add(row[j] - new_val)
                
            # Check internal row conflicts
            # The new diffs must not conflict with other diffs in the SAME row that don't involve pos
            # And must not conflict with OTHER rows.
            
            # Diffs in same row not involving pos:
            other_diffs_in_row = current_diffs[r_idx] - old_diffs_pos
            
            # If new_diffs_pos intersects with other_diffs_in_row, invalid
            if new_diffs_pos & other_diffs_in_row:
                continue
                
            # Check against other rows
            # The set of diffs used by other rows is: all_used - current_diffs[r_idx]
            other_rows_diffs = all_used - current_diffs[r_idx]
            
            if new_diffs_pos & other_rows_diffs:
                continue
                
            # Move is valid
            # Apply move
            row[pos] = new_val
            current_diffs[r_idx] = (current_diffs[r_idx] - old_diffs_pos) | new_diffs_pos
            all_used = (all_used - old_diffs_pos) | new_diffs_pos
            
            new_obj = max(max(r) for r in current_rows)
            
            if new_obj < current_obj:
                best_obj = new_obj
                best_rows = [r[:] for r in current_rows]
                current_obj = new_obj
            else:
                # Accept with probability exp(-(new_obj - current_obj)/T)
                # Note: objective is integer, so diff is small
                if random.random() < pow(2.71828, -(new_obj - current_obj) / T):
                    current_obj = new_obj
                else:
                    # Reject: undo move
                    row[pos] = old_val
                    current_diffs[r_idx] = (current_diffs[r_idx] - new_diffs_pos) | old_diffs_pos
                    all_used = (all_used - new_diffs_pos) | old_diffs_pos
                    
        elif move_type < 0.7:
            # Row Rebuild
            r_idx = random.randint(0, n_rows - 1)
            # Diffs used by other rows
            other_diffs = set()
            for i in range(n_rows):
                if i != r_idx:
                    other_diffs |= current_diffs[i]
            
            new_row = rebuild_row(current_rows, r_idx, other_diffs, randomize=True)
            if new_row is None:
                continue
                
            new_diffs = get_diffs(new_row)
            
            # Check if it conflicts with other rows (it shouldn't by construction, but double check)
            if new_diffs & other_diffs:
                continue
                
            # Calculate new objective
            # We need to update current_rows temporarily to check max
            old_row = current_rows[r_idx]
            current_rows[r_idx] = new_row
            new_obj = max(max(r) for r in current_rows)
            
            if new_obj < current_obj:
                best_obj = new_obj
                best_rows = [r[:] for r in current_rows]
                current_obj = new_obj
                current_diffs[r_idx] = new_diffs
                all_used = (all_used - get_diffs(old_row)) | new_diffs
            else:
                if random.random() < pow(2.71828, -(new_obj - current_obj) / T):
                    current_obj = new_obj
                    current_diffs[r_idx] = new_diffs
                    all_used = (all_used - get_diffs(old_row)) | new_diffs
                else:
                    # Reject
                    current_rows[r_idx] = old_row
                    # current_diffs and all_used remain unchanged
            
        else:
            # Perturbation: Randomly reset one row to a greedy construction with high randomization
            r_idx = random.randint(0, n_rows - 1)
            other_diffs = set()
            for i in range(n_rows):
                if i != r_idx:
                    other_diffs |= current_diffs[i]
            
            # Try a few times to find a valid row
            for attempt in range(5):
                new_row = greedy_construct(other_diffs, randomize=True)
                if new_row:
                    new_diffs = get_diffs(new_row)
                    if not (new_diffs & other_diffs):
                        old_row = current_rows[r_idx]
                        current_rows[r_idx] = new_row
                        new_obj = max(max(r) for r in current_rows)
                        
                        if new_obj < current_obj:
                            best_obj = new_obj
                            best_rows = [r[:] for r in current_rows]
                            current_obj = new_obj
                            current_diffs[r_idx] = new_diffs
                            all_used = (all_used - get_diffs(old_row)) | new_diffs
                        else:
                            if random.random() < pow(2.71828, -(new_obj - current_obj) / T):
                                current_obj = new_obj
                                current_diffs[r_idx] = new_diffs
                                all_used = (all_used - get_diffs(old_row)) | new_diffs
                            else:
                                current_rows[r_idx] = old_row
                        break

        T *= cooling
        if T < 1.0:
            T = 1.0

    # Final verification
    if best_rows and is_valid_solution(best_rows):
        return best_rows
    else:
        # Fallback to a known valid solution (greedy)
        rows = []
        used = set()
        for r in range(n_rows):
            row = greedy_construct(used, randomize=False)
            rows.append(row)
            used |= get_diffs(row)
        return rows

