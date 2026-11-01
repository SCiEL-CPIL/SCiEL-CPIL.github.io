import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 138s to be safe.
    deadline = start_time + 138.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        """Check if rows are valid (strictly increasing, start at 0, unique diffs)."""
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def build_diff_index(rows):
        """Build dict: diff -> (row_idx, i, j) for all diffs. Returns None if conflict."""
        index = {}
        for ri, row in enumerate(rows):
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in index:
                        return None
                    index[d] = (ri, i, j)
        return index

    def try_move(rows, idx, pos, new_val, diff_index):
        """
        Try changing rows[idx][pos] to new_val.
        Returns True if successful and applies the change, False otherwise.
        Modifies rows and diff_index in place.
        """
        row = rows[idx]
        old_val = row[pos]
        if new_val == old_val:
            return False
        
        # Check strict increasing
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False

        n = len(row)
        affected = []  # list of (old_diff, new_diff, i, j)
        
        # Collect affected differences
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d_old = old_val - row[i]
                d_new = new_val - row[i]
                affected.append((d_old, d_new, i, pos))
            else:
                d_old = row[i] - old_val
                d_new = row[i] - new_val
                affected.append((d_old, d_new, pos, i))

        # Check validity of old diffs in index (sanity check) and new diffs for conflicts
        for d_old, d_new, i, j in affected:
            if d_old not in diff_index:
                return False
            stored = diff_index[d_old]
            if stored != (idx, i, j):
                return False
            
            if d_new <= 0:
                return False
            if d_new in diff_index:
                stored = diff_index[d_new]
                # If the diff is already used by another pair, it's a conflict
                if stored != (idx, i, j): 
                    return False

        # Apply changes
        row[pos] = new_val
        for d_old, d_new, i, j in affected:
            del diff_index[d_old]
            diff_index[d_new] = (idx, i, j)
            
        return True

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def greedy_init(seed=None, rand_prob=0.1):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 200 
                while x <= limit and len(candidates) < 20:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def rebuild_row(used_other, max_val_limit, rng, jitter=0.0):
        """
        Rebuild a single row given differences already used by other rows.
        Returns the row list or None if failed.
        """
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            # Search window: from x+1 to max_val_limit
            # We prefer smaller values to keep scope low
            search_end = max_val_limit
            # If max_val_limit is None or very large, cap search for efficiency in early stages? 
            # No, we need to find *some* valid row. But we want it small.
            # Let's scan sequentially.
            
            # Optimization: If we are stuck, we might need to go higher.
            # But for 'rebuild' we usually have a target scope in mind.
            
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    # Apply jitter: skip small values sometimes to explore different structures?
                    # Actually, for minimization, we want the *smallest* valid value.
                    # Jitter is useful for *perturbation* (finding a different valid row), 
                    # not for minimization. 
                    # However, if we always pick the smallest, we might get stuck in a local structure.
                    # So, if jitter > 0, we might skip the first few candidates.
                    if jitter > 0 and rng.random() < jitter and trial_x < search_end - 10:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            
            if not found:
                return None
            x = row[-1]
        return row

    def local_search(rows, time_limit):
        """
        Iterated Local Search with:
        1. Single element moves (using diff_index for speed).
        2. Row reconstruction (perturbation).
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        diff_index = build_diff_index(best_rows)
        
        if diff_index is None:
            return best_rows, best_scope

        rng = random.Random(123)
        stuck_counter = 0

        while time.time() < start + time_limit:
            # 1. Try to improve scope via single element moves
            improved = False
            
            # Identify the element with the maximum value to target for reduction
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx

            # Try to reduce this specific element
            if max_c_idx > 0:
                curr_val = max_val
                prev_val = best_rows[max_r_idx][max_c_idx - 1]
                
                # Try values from curr_val - 1 down to prev_val + 1
                for new_val in range(curr_val - 1, prev_val, -1):
                    if try_move(best_rows, max_r_idx, max_c_idx, new_val, diff_index):
                        improved = True
                        new_scope = get_scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            stuck_counter = 0
                        break
            
            # If we didn't improve the global max, try other elements that might be reduced
            # to allow future improvements, or just general descent.
            if not improved:
                # Pick a random element to try to reduce
                r_idx = rng.randint(0, n_rows - 1)
                c_idx = rng.randint(1, row_len - 1)
                curr_val = best_rows[r_idx][c_idx]
                prev_val = best_rows[r_idx][c_idx - 1]
                
                for new_val in range(curr_val - 1, prev_val, -1):
                    if try_move(best_rows, r_idx, c_idx, new_val, diff_index):
                        improved = True
                        new_scope = get_scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            stuck_counter = 0
                        break
            
            if improved:
                continue
            
            # Stuck: Perturb via Row Reconstruction
            stuck_counter += 1
            
            if stuck_counter > 5:
                # Pick a random row to reconstruct
                r_idx = rng.randint(0, n_rows - 1)
                
                # Calculate used diffs by other rows
                used_other = set()
                for j in range(n_rows):
                    if j != r_idx:
                        used_other |= row_diffs_set(best_rows[j])
                
                # Try to rebuild the row with a scope limit
                # We want to keep the scope low, so limit to current best_scope
                # Sometimes allow slightly higher to escape? No, we want to minimize.
                # If we can't rebuild within best_scope, we might need to expand?
                # But expanding scope is bad. We only accept if valid.
                # Let's try to rebuild within best_scope first.
                
                new_row = None
                # Try a few times with different random seeds/jitters for diversity
                for attempt in range(10):
                    # Jitter helps find different valid rows if the greedy one fails
                    j = 0.0 if attempt == 0 else 0.1 + 0.05 * attempt
                    new_row = rebuild_row(used_other, best_scope, rng, jitter=j)
                    if new_row is not None:
                        break
                
                if new_row is not None:
                    # Check if this new row is actually better or equal
                    # It must be valid by construction (unique diffs within row, no conflict with others)
                    # We need to update diff_index
                    # Remove old row diffs
                    old_row = best_rows[r_idx]
                    for i in range(len(old_row)):
                        for j in range(i+1, len(old_row)):
                            d = old_row[j] - old_row[i]
                            if d in diff_index:
                                del diff_index[d]
                    
                    # Add new row diffs
                    best_rows[r_idx] = new_row
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            diff_index[d] = (r_idx, i, j)
                    
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    stuck_counter = 0
                else:
                    # If we can't rebuild within best_scope, we are really stuck.
                    # Try a looser constraint? If we allow higher scope, we might find a path back down.
                    # But increasing scope is risky. 
                    # Let's try a standard random move perturbation instead.
                    pass

        return best_rows, best_scope

    def high_precision_polish(rows, time_limit):
        """
        Systematically try to reduce every element to the smallest possible value.
        """
        start = time.time()
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        diff_index = build_diff_index(current_rows)
        
        if diff_index is None:
            return current_rows, current_scope

        improved = True
        while improved and time.time() < start + time_limit:
            improved = False
            # Iterate through all elements in order of magnitude (largest first)
            # or just row by row. Row by row is fine.
            for r_idx in range(n_rows):
                for c_idx in range(row_len - 1, 0, -1): # From right to left
                    if time.time() > start + time_limit:
                        return current_rows, current_scope
                    
                    curr_val = current_rows[r_idx][c_idx]
                    prev_val = current_rows[r_idx][c_idx - 1]
                    
                    # Try to reduce curr_val
                    for new_val in range(curr_val - 1, prev_val, -1):
                        if try_move(current_rows, r_idx, c_idx, new_val, diff_index):
                            improved = True
                            new_scope = get_scope(current_rows)
                            if new_scope < current_scope:
                                current_scope = new_scope
                            break # Move to next element immediately after improvement
                            
        return current_rows, current_scope

    # Phase 1: Diverse Initialization
    best_rows = None
    best_scope = float('inf')
    
    num_inits = 10
    init_time_limit = 5.0
    start_init = time.time()
    
    for i in range(num_inits):
        if time.time() > start_init + init_time_limit:
            break
        rows = greedy_init(rand_prob=0.05 + 0.01*i, seed=i)
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    if best_rows is None:
        # Fallback
        best_rows = greedy_init(seed=0)
        best_scope = get_scope(best_rows)

    # Phase 2: Local Search Refinement
    # Allocate time based on remaining budget
    remaining = deadline - time.time() - 10 # 10s safety margin
    local_time = remaining * 0.6
    
    if local_time > 5 and best_rows:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 3: High-precision Polish
    remaining = deadline - time.time() - 5
    polish_time = remaining * 0.4
    
    if polish_time > 2 and best_rows:
        polished_rows, polished_scope = high_precision_polish(best_rows, polish_time)
        if validate(polished_rows):
            if polished_scope < best_scope:
                best_rows = polished_rows
                best_scope = polished_scope

    # Final check
    if not validate(best_rows):
        # Should not happen, but safe fallback
        best_rows = greedy_init(seed=999)
        best_scope = get_scope(best_rows)

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
