import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

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

    def greedy_init(rand_prob=0.15, window=30):
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
                limit = 300
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
            
            # Let's build a set of all diffs currently in the system.
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
                
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, len(row))]
                
                # Check for duplicates within new diffs
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                
                # If we pass all checks, this new_val is valid.
                row[max_c_idx] = new_val
                found = True
                if new_val < best_scope:
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                break
            
            if not found:
                # If we can't reduce the current max, try a random swap on a non-max element to "jolt" the system.
                success_in_step = False
                r_idx = random.randint(0, n_rows-1)
                c_idx = random.randint(1, row_len-1)
                
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx-1]
                next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 100
                
                # Re-compute fixed diffs
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
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len-1 else curr_val + 50
                
                for _ in range(30):
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
                    
                    row[c_idx] = cand
                    success_in_step = True
                    break
                
                if not success_in_step:
                    # Stuck, break to save time
                    pass

        return best_rows, best_scope

    # Phase 1: Get a baseline
    best_rows = greedy_init(0.15, 30)
    best_scope = get_scope(best_rows)
    
    # Phase 2: Local Search Refinement
    local_time = 60.0
    if time.time() > deadline - 70:
        local_time = deadline - time.time() - 10
        
    if local_time > 10:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 3: Backtracking to try to beat the local search result
    current_target = best_scope - 1
    seed_counter = 0

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
            scan_limit = min(upper, lower + 60)
            for val in range(lower, scan_limit + 1):
                valid = True
                for i in range(pos):
                    d = val - rows[row_idx][i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(val)
                    if len(candidates) >= 12:
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
                        if len(candidates) >= 12:
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
                    break
            else:
                break
        else:
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
