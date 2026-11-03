import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    best_max = float('inf')
    best_rows = None

    # --- Greedy Baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_max = max(max(r) for r in best_rows)

    # --- Incremental SA with Diff Counting ---
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def sa_optimize(initial_rows, time_limit, bound):
        rows = [r[:] for r in initial_rows]
        n = n_rows
        L = row_len

        # Precompute diffs per row
        row_diffs = [get_diffs(r) for r in rows]
        
        # Global diff counts
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        # Cost: sum of (count > 1 ? count-1 : 0) * 1000 + max_val
        def compute_cost():
            collisions = 0
            for d, c in diff_counts.items():
                if c > 1:
                    collisions += c - 1
            max_val = max(r[-1] for r in rows)
            return collisions * 10000 + max_val

        current_cost = compute_cost()
        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = max(r[-1] for r in best_local_rows)

        T = 50.0
        cooling = 0.9995
        start_time = time.time()

        while time.time() - start_time < time_limit:
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, L - 1)
            
            row = rows[r_idx]
            old_val = row[i_idx]
            
            # Determine range
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < L - 1 else bound + 10
            upper = min(upper, bound + 20)
            
            if lower > upper:
                continue
                
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue
            
            # Remove old diffs for this row
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            # Apply change
            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs
            
            # Add new diffs
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_cost = compute_cost()
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_rows = [r[:] for r in rows]
                    best_local_max = max(r[-1] for r in best_local_rows)
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if random.random() < 0.5:
                        current_cost = new_cost
                    else:
                        # Revert
                        row[i_idx] = old_val
                        for d in new_diffs:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs
                else:
                    if delta < 1000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    
                    if random.random() < p:
                        current_cost = new_cost
                    else:
                        # Revert
                        row[i_idx] = old_val
                        for d in new_diffs:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs
            
            T *= cooling
            if T < 0.1:
                T = 0.1

        return best_local_rows, best_local_max, best_local_cost

    # --- CSP Backtracking with Forward Checking ---
    def csp_construct(target_M, time_limit):
        used = set()
        rows = []
        start_time = time.time()
        
        def dfs(row_idx):
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Try to construct row row_idx
            # We need k+1 numbers: 0 = x0 < x1 < ... < xk <= target_M
            # Differences must be distinct within row and not in used.
            
            # Heuristic: Try to pick the row greedily but with backtracking on the last few elements?
            # Or just random restarts with greedy completion?
            
            # Let's try a simple backtracking on the elements x1..xk
            # This might be slow if target_M is large.
            
            # Alternative: Generate a random valid row within [0, target_M] that fits `used`.
            # If found, add to used and recurse.
            # If not found after many tries, return False.
            
            attempts = 0
            while attempts < 50:
                attempts += 1
                # Pick k random points from 1..target_M
                if target_M < k:
                    return False
                
                pts = random.sample(range(1, target_M + 1), k)
                pts.sort()
                row = [0] + pts
                
                # Check internal diffs
                diffs = get_diffs(row)
                if len(set(diffs)) != len(diffs):
                    continue
                
                # Check external diffs
                if any(d in used for d in diffs):
                    continue
                
                # Valid row found
                rows.append(row)
                used.update(diffs)
                
                if dfs(row_idx + 1):
                    return True
                else:
                    # Backtrack
                    rows.pop()
                    for d in diffs:
                        used.discard(d)
            
            return False
        
        if dfs(0):
            return rows
        return None

    # Main Loop
    while time.time() < deadline - 5.0:
        # 1. Try CSP to find a valid solution with smaller scope
        # Start from current best_max and go down
        for target in range(best_max, best_max - 20, -1):
            if time.time() > deadline - 10.0:
                break
            csp_time = min(2.0, deadline - time.time() - 10.0)
            if csp_time <= 0:
                break
            
            result = csp_construct(target, csp_time)
            if result is not None:
                new_max = max(r[-1] for r in result)
                if new_max < best_max:
                    best_rows = result
                    best_max = new_max
                    break # Found a valid one, break target loop
        
        # 2. Polish with SA
        sa_time = min(5.0, deadline - time.time() - 2.0)
        if sa_time <= 0:
            break
        
        # Initialize near best solution
        init_rows = [r[:] for r in best_rows]
        # Add slight randomization
        for _ in range(3):
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            row = init_rows[r_idx]
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < row_len - 1 else best_max + 5
            if lower <= upper:
                init_rows[r_idx][i_idx] = random.randint(lower, upper)
        
        local_rows, local_max, local_cost = sa_optimize(init_rows, sa_time, best_max)
        
        if local_cost < 10000: # Valid
            if local_max < best_max:
                best_rows = local_rows
                best_max = local_max

    return best_rows
