import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0

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

    def greedy_init(rand_prob=0.1, window=20):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 200 
                while x <= limit and len(candidates) < window:
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

    def local_search(rows, time_limit):
        """
        Structural refinement: Targeted descent to minimize the global maximum (scope).
        We repeatedly identify the largest element in the entire configuration.
        We attempt to replace it with a smaller valid value.
        If successful, the scope might decrease, or at least the distribution tightens.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        # Precompute current diffs for quick checking? 
        # Actually, checking a single value change against all other diffs is O(total_diffs) which is fine for small N.
        # But we can be smarter.
        
        # We'll maintain a set of all used diffs for the current state to check validity quickly.
        # However, when we change one element, we remove its diffs and add new ones.
        
        while time.time() < start + time_limit:
            # Find the index of the global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            # We don't want to change the first element (0)
            if max_c_idx == 0:
                break
            
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            
            # Try to find a smaller value for this position
            # Range: (prev_val, current_val)
            # We need to check if this new value creates any duplicate diffs with:
            # 1. Other elements in the same row
            # 2. Any diffs in other rows (that don't involve this element)
            
            # Let's collect all "fixed" diffs (diffs not involving the element we're changing)
            # Diffs involving element (r, c) are:
            # (val - best_rows[r][i]) for i < c
            # (best_rows[r][j] - val) for j > c  <-- Wait, differences are defined as a_j - a_i with i<j.
            # So diffs are always positive.
            # For element at index c:
            # It forms diffs with i < c: val - row[i]
            # It forms diffs with j > c: row[j] - val
            
            # Let's build a set of all diffs currently in the system.
            # Then remove the ones involving the target element.
            # Then check candidates against the remaining set + new internal diffs.
            
            all_diffs = set()
            for r in best_rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        all_diffs.add(r[j] - r[i])
            
            # Diffs to remove (involving target)
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, len(row)):
                remove_set.add(row[j] - current_val)
            
            # Fixed diffs are all_diffs - remove_set
            fixed_diffs = all_diffs - remove_set
            
            found = False
            # Try values from current_val - 1 down to prev_val + 1
            # We can randomize the order slightly or just go down.
            # Going down is good for minimizing scope immediately.
            # But maybe skipping some is faster? No, we want the smallest valid.
            
            for new_val in range(current_val - 1, prev_val, -1):
                # Check validity against fixed_diffs
                valid = True
                # Check diffs with left neighbors
                for i in range(max_c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Check diffs with right neighbors
                for j in range(max_c_idx + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check internal consistency? 
                # The new diffs are (new_val - row[i]) and (row[j] - new_val).
                # Do any of these collide with each other?
                # i.e., is (new_val - row[i]) == (row[j] - new_val) for some i<j?
                # Or (new_val - row[i1]) == (new_val - row[i2])? No, row is increasing.
                # Or (row[j1] - new_val) == (row[j2] - new_val)? No.
                # So we just need to ensure no new diff equals another new diff?
                # Actually, if d1 = new_val - row[i] and d2 = row[j] - new_val.
                # If d1 == d2, then that diff appears twice in this row.
                # We must check this.
                
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, len(row))]
                
                # Check for duplicates within new diffs
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                
                # If we pass all checks, this new_val is valid.
                # Update the row
                row[max_c_idx] = new_val
                found = True
                # Update best_scope if this was the global max
                if new_val < best_scope:
                     # Recalculate scope
                     new_scope = get_scope(best_rows)
                     if new_scope < best_scope:
                         best_scope = new_scope
                break
            
            if not found:
                # If we can't reduce the current max, maybe we should try another strategy?
                # For now, if we can't reduce the global max, we might be stuck in a local minimum
                # where the global max is "pinned".
                # To escape, we could try reducing the *second* largest, or randomizing a non-max element.
                # Let's try to reduce the second largest element if we can't reduce the largest.
                
                # Find second largest
                sorted_indices = sorted(
                    [(best_rows[r][c], r, c) for r in range(n_rows) for c in range(row_len) if c > 0],
                    reverse=True
                )
                
                # Try the top 5 largest elements that are not the global max (or including it if we failed)
                # Actually, if we failed on the max, trying the second max might allow the max to be reduced later.
                
                success_in_step = False
                for val, r_idx, c_idx in sorted_indices[:5]:
                    if time.time() > start + time_limit:
                        break
                    if val == 0: continue
                    
                    # Same logic as above but for this specific element
                    # This is getting expensive. Let's just restart local search with a different seed/randomization?
                    # Or simpler: if we can't improve the max, try to swap a non-max element to a random valid value to "jolt" the system.
                    
                    # Let's try a simple random swap on a non-max element.
                    r_idx = random.randint(0, n_rows-1)
                    c_idx = random.randint(1, row_len-1)
                    
                    curr_val = best_rows[r_idx][c_idx]
                    prev_v = best_rows[r_idx][c_idx-1]
                    next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 100
                    
                    # Try a random value in (prev_v, curr_val) U (curr_val, next_v)
                    # Just try to change it to something else valid.
                    
                    # Re-compute fixed diffs (expensive but safe)
                    all_diffs = set()
                    for r in best_rows:
                        for i in range(len(r)):
                            for j in range(i+1, len(r)):
                                all_diffs.add(r[j] - r[i])
                    
                    remove_set = set()
                    row = best_rows[r_idx]
                    for i in range(c_idx):
                        remove_set.add(curr_val - row[i])
                    for j in range(c_idx + 1, len(row)):
                        remove_set.add(row[j] - curr_val)
                    
                    fixed_diffs = all_diffs - remove_set
                    
                    # Try a few random candidates
                    candidates = []
                    # Sample from range
                    low = prev_v + 1
                    high = next_v - 1 if c_idx < row_len-1 else curr_val + 50
                    
                    # We want to try values close to current to maintain structure, or random.
                    for _ in range(20):
                        if low > high: break
                        cand = random.randint(low, high)
                        if cand == curr_val: continue
                        
                        valid = True
                        for i in range(c_idx):
                            d = cand - row[i]
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid: continue
                        
                        for j in range(c_idx + 1, len(row)):
                            d = row[j] - cand
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid: continue
                        
                        new_left = [cand - row[i] for i in range(c_idx)]
                        new_right = [row[j] - cand for j in range(c_idx+1, len(row))]
                        if len(set(new_left + new_right)) != len(new_left + new_right):
                            continue
                        
                        # Apply
                        row[c_idx] = cand
                        success_in_step = True
                        break
                    
                    if success_in_step:
                        break
                
                if not success_in_step:
                    # Stuck?
                    # If we can't move the max and can't jolt, we might be done.
                    # Or we can try to restart the whole local search with a different random seed?
                    # For now, break to save time.
                    pass

        return best_rows, best_scope

    # Phase 1: Get a baseline
    best_rows = greedy_init(0.1, 20)
    best_scope = get_scope(best_rows)
    
    # Phase 2: Local Search Refinement
    # Allocate a significant chunk of time to local search to tighten the solution.
    local_time = 60.0
    if time.time() > deadline - 70:
        local_time = deadline - time.time() - 10
        
    if local_time > 10:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope
        else:
            # Fallback to original if validation failed (shouldn't happen)
            pass

    # Phase 3: Backtracking to try to beat the local search result
    # We use the local search result as a guide for the target scope.
    # We try to find a solution with scope < best_scope.
    
    current_target = best_scope - 1
    seed_counter = 0
    
    # Define backtracking search
    def backtracking_search(n_rows, k, target_scope, time_limit, seed=0):
        """
        Attempts to find a valid configuration with max scope <= target_scope.
        Uses randomized backtracking.
        """
        random.seed(seed)
        start = time.time()
        
        rows = [[0] * row_len for _ in range(n_rows)]
        used_diffs = set()
        
        def solve(row_idx, pos):
            if time.time() > start + time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == row_len:
                return solve(row_idx + 1, 1)
            
            prev_val = rows[row_idx][pos - 1]
            lower = prev_val + 1
            upper = target_scope
            
            if lower > upper:
                return False
            
            candidates = []
            scan_limit = min(upper, lower + 50)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 10:
                        break
            
            if not candidates:
                for val in range(scan_limit + 1, upper + 1):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                        if len(candidates) >= 10:
                            break
            
            if not candidates:
                return False
            
            random.shuffle(candidates)
            
            for val in candidates:
                new_diffs = []
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    new_diffs.append(d)
                    used_diffs.add(d)
                
                rows[row_idx][pos] = val
                
                if solve(row_idx, pos + 1):
                    return True
                
                rows[row_idx][pos] = 0
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return False

        if solve(0, 1):
            return [r[:] for r in rows]
        return None

    # Run backtracking attempts
    while time.time() < deadline - 5:
        if current_target < 112:
            break
            
        # Adaptive time for backtracking
        # If we are close to the deadline, use less time.
        search_time = 5.0
        if time.time() > deadline - 15:
            search_time = deadline - time.time() - 2
            
        if search_time < 1.0:
            break

        result = backtracking_search(n_rows, k, current_target, search_time, seed=seed_counter)
        seed_counter += 1
        
        if result:
            if validate(result):
                new_scope = get_scope(result)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = result
                    current_target = best_scope - 1
                else:
                    # Found a valid solution but scope didn't improve? 
                    # This implies the search found a solution with scope <= target, 
                    # but maybe target was not tight enough or we made a mistake.
                    # If new_scope <= current_target and current_target < best_scope, it should improve.
                    # If it doesn't, something is off.
                    break
            else:
                # Invalid result from backtracking? Should not happen.
                break
        else:
            # Failed to find solution with current_target
            # Try one retry with different seed
            if time.time() < deadline - 10:
                result_retry = backtracking_search(n_rows, k, current_target, 5.0, seed=seed_counter)
                seed_counter += 1
                if result_retry and validate(result_retry):
                    new_scope = get_scope(result_retry)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = result_retry
                        current_target = best_scope - 1
                    else:
                        break
                else:
                    break
            else:
                break

    return best_rows

if __name__ == "__main__":
    # Simple test
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    # Verify
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
