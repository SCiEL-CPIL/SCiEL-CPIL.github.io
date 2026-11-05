import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(scope_limit=200):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                for x in range(last_val + 1, scope_limit + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def sa_optimize(initial_rows, time_limit):
        rows = [r[:] for r in initial_rows]
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Use bytearray for speed if scope is small
        max_diff_estimate = best_scope + 50
        use_bitset = max_diff_estimate < 512
        
        if use_bitset:
            used = bytearray(max_diff_estimate + 1)
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d <= max_diff_estimate:
                            used[d] = 1
        else:
            used = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used.add(row[j] - row[i])

        t_end = time.time() + time_limit
        temp = max(1.0, current_scope / 5.0)
        cooling = 0.9999
        
        steps = 0
        while time.time() < t_end:
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            old_val = rows[row_idx][pos]
            min_val = rows[row_idx][pos - 1] + 1
            max_val = rows[row_idx][pos + 1] - 1 if pos < k else best_scope + 20
            
            if min_val > max_val:
                temp *= cooling
                continue
                
            # Generate new value
            if random.random() < 0.6:
                # Try to shrink
                if old_val > min_val:
                    new_val = random.randint(min_val, old_val - 1)
                else:
                    new_val = old_val
            else:
                new_val = random.randint(min_val, max_val)
                
            if new_val == old_val:
                temp *= cooling
                continue

            # Check validity
            valid = True
            new_diffs = []
            
            if use_bitset:
                # Remove old
                for i in range(k + 1):
                    if i != pos:
                        d = abs(old_val - rows[row_idx][i])
                        if d <= max_diff_estimate:
                            used[d] = 0
                # Check new
                for i in range(k + 1):
                    if i != pos:
                        d = abs(new_val - rows[row_idx][i])
                        if d > max_diff_estimate:
                            valid = False
                            break
                        if used[d]:
                            valid = False
                            break
                        new_diffs.append(d)
            else:
                old_diffs = set()
                for i in range(k + 1):
                    if i != pos:
                        d = abs(old_val - rows[row_idx][i])
                        old_diffs.add(d)
                
                for d in old_diffs:
                    used.discard(d)
                    
                for i in range(k + 1):
                    if i != pos:
                        d = abs(new_val - rows[row_idx][i])
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)

            if not valid:
                # Revert
                if use_bitset:
                    for i in range(k + 1):
                        if i != pos:
                            d = abs(old_val - rows[row_idx][i])
                            if d <= max_diff_estimate:
                                used[d] = 1
                else:
                    for d in old_diffs:
                        used.add(d)
                temp *= cooling
                continue

            # Apply
            rows[row_idx][pos] = new_val
            if use_bitset:
                for d in new_diffs:
                    used[d] = 1
            else:
                for d in new_diffs:
                    used.add(d)
            
            new_scope = max(rows[row_idx][-1], max(r[-1] for r in rows if r is not rows[row_idx]))
            
            # Acceptance
            if new_scope < current_scope:
                accept = True
            elif new_scope == current_scope:
                accept = random.random() < 0.5
            else:
                delta = new_scope - current_scope
                accept = random.random() < pow(2.71828, -delta / temp)
                
            if accept:
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Revert
                rows[row_idx][pos] = old_val
                if use_bitset:
                    for d in new_diffs:
                        used[d] = 0
                    for i in range(k + 1):
                        if i != pos:
                            d = abs(old_val - rows[row_idx][i])
                            if d <= max_diff_estimate:
                                used[d] = 1
                else:
                    for d in new_diffs:
                        used.discard(d)
                    for i in range(k + 1):
                        if i != pos:
                            d = abs(old_val - rows[row_idx][i])
                            used.add(d)
                            
            temp *= cooling
            steps += 1

        return best_rows, best_scope

    def polish(rows, time_limit):
        rows = [r[:] for r in rows]
        current_scope = get_scope(rows)
        t_end = time.time() + time_limit
        
        improved = True
        while improved and time.time() < t_end:
            improved = False
            # Try to shrink each element
            for r_idx in range(n_rows):
                for p in range(k, 0, -1):
                    if time.time() >= t_end: break
                    row = rows[r_idx]
                    old_val = row[p]
                    min_val = row[p-1] + 1
                    max_val = row[p+1] - 1 if p < k else current_scope
                    
                    # Try values from old_val-1 down to min_val
                    for new_val in range(old_val - 1, min_val - 1, -1):
                        # Check validity
                        valid = True
                        # We need to check against all other diffs
                        # This is O(N*K^2) per check, which is slow.
                        # Optimization: Only check diffs involving new_val
                        # And ensure they don't conflict with existing diffs NOT involving old_val
                        # This is complex. 
                        # Given the small size, a simple check is fine if we limit attempts.
                        
                        # Let's just try the immediate decrement first.
                        if new_val < old_val:
                             # Quick check
                             pass
                        break # Only try one step for now to keep it fast? 
                             # Actually, let's just try old_val - 1.
                    
                    # Simplified Polish: Just try to decrement by 1 if valid.
                    new_val = old_val - 1
                    if new_val < min_val: continue
                    
                    # Check validity of new_val
                    # Remove old diffs, add new diffs, check conflicts.
                    # Reusing SA logic here is redundant. 
                    # Let's just run a few SA steps with very low temp?
                    # No, let's do a simple check.
                    
                    # For speed, we will skip the complex polish and rely on SA.
                    # SA with shrink bias effectively polishes.
                    pass
        
        return rows, current_scope

    # Main Logic
    best_rows = None
    best_scope = float('inf')
    
    # 1. Greedy
    rows = greedy_construct(200)
    if rows:
        s = get_scope(rows)
        if s < best_scope:
            best_scope = s
            best_rows = rows
            
    # 2. ILS
    while time.time() < deadline - 10.0:
        if best_rows:
            start_rows = [r[:] for r in best_rows]
            # Perturb
            for _ in range(2):
                r = random.randint(0, n_rows-1)
                p = random.randint(1, k)
                min_v = start_rows[r][p-1] + 1
                max_v = start_rows[r][p+1] - 1 if p < k else best_scope + 20
                if min_v <= max_v:
                    start_rows[r][p] = random.randint(min_v, max_v)
        else:
            start_rows = greedy_construct(200)
            if not start_rows: continue
            
        time_limit = min(15.0, deadline - time.time() - 5.0)
        if time_limit < 1.0: break
        
        res_rows, res_scope = sa_optimize(start_rows, time_limit)
        
        if res_scope < best_scope:
            best_scope = res_scope
            best_rows = res_rows

    # 3. Final SA run on best
    if best_rows and time.time() < deadline - 2.0:
        final_rows, final_scope = sa_optimize(best_rows, deadline - time.time() - 1.0)
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = final_rows

    # Verify
    if best_rows and not is_valid(best_rows):
        # Fallback
        return greedy_construct(500)
        
    return best_rows
