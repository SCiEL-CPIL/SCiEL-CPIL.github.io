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

    # --- Helper: get all pairwise differences of a row ---
    def get_diffs(row):
        diffs = []
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                diffs.append(row[j] - row[i])
        return diffs

    # --- Build a row with minimal max given used_diffs and a bound ---
    def build_row_minmax(used_diffs, max_allowed, time_limit=None):
        if time_limit is not None and time.time() > time_limit:
            return None, float('inf')
        
        row = [0]
        best_found = None
        
        def dfs(pos):
            nonlocal best_found
            if time_limit is not None and time.time() > time_limit:
                return
            if best_found is not None and row[-1] >= best_found[-1]:
                return
            
            if pos == row_len:
                best_found = list(row)
                return
            
            for v in range(row[-1] + 1, max_allowed + 1):
                if best_found is not None and v >= best_found[-1]:
                    break
                
                valid = True
                for i in range(pos):
                    d = v - row[i]
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    for i in range(pos):
                        d = v - row[i]
                        for j in range(i + 1, pos):
                            if d == row[j] - row[i]:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    row.append(v)
                    dfs(pos + 1)
                    row.pop()
                    if best_found is not None:
                        return
        
        dfs(1)
        if best_found is not None:
            return best_found, best_found[-1]
        return None, float('inf')

    # --- Try to construct all rows with max <= M ---
    def try_construct_with_bound(M, time_limit):
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            if time_limit is not None and time.time() > time_limit:
                return None
            
            row, row_max = build_row_minmax(used_diffs, M, time_limit)
            if row is None:
                return None
            
            for d in get_diffs(row):
                used_diffs.add(d)
            rows.append(row)
        
        return rows

    # --- SA Polish ---
    def sa_polish(init_rows, current_max, time_limit):
        rows_copy = [r[:] for r in init_rows]
        row_diffs = [get_diffs(r) for r in rows_copy]
        
        diff_counts = {}
        for rd in row_diffs:
            for d in rd:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        def compute_cost():
            collisions = 0
            for d, c in diff_counts.items():
                if c > 1:
                    collisions += c - 1
            max_val = max(r[-1] for r in rows_copy)
            return collisions * 100000 + max_val
        
        current_cost = compute_cost()
        best_cost = current_cost
        best_rows_copy = [r[:] for r in rows_copy]
        best_max_local = max(r[-1] for r in best_rows_copy)
        
        T = 100.0
        cooling = 0.999
        start_time = time.time()
        
        while time.time() < time_limit:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            row = rows_copy[r_idx]
            old_val = row[i_idx]
            
            lower = row[i_idx - 1] + 1
            upper = row[i_idx + 1] - 1 if i_idx < row_len - 1 else current_max + 20
            upper = min(upper, current_max + 30)
            
            if lower > upper:
                continue
            
            new_val = random.randint(lower, upper)
            if new_val == old_val:
                continue
            
            old_diffs = row_diffs[r_idx]
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            row[i_idx] = new_val
            new_diffs = get_diffs(row)
            row_diffs[r_idx] = new_diffs
            
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_cost = compute_cost()
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rows_copy = [r[:] for r in rows_copy]
                    best_max_local = max(r[-1] for r in best_rows_copy)
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if random.random() < 0.5:
                        current_cost = new_cost
                    else:
                        row[i_idx] = old_val
                        for d in new_diffs:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs
                else:
                    if delta < 5000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    
                    if random.random() < p:
                        current_cost = new_cost
                    else:
                        row[i_idx] = old_val
                        for d in new_diffs:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        for d in old_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        row_diffs[r_idx] = old_diffs
            
            T *= cooling
            if T < 0.5:
                T = 0.5
        
        if best_cost < 100000:
            all_diffs = []
            for r in best_rows_copy:
                all_diffs.extend(get_diffs(r))
            if len(all_diffs) == len(set(all_diffs)):
                return best_rows_copy, best_max_local
        
        return None, float('inf')

    # --- Main loop: iterative deepening ---
    current_bound = best_max
    no_improve_count = 0
    
    while time.time() < deadline - 15.0:
        new_bound = current_bound - 1
        if new_bound < 10:
            break
        
        result = try_construct_with_bound(new_bound, deadline - 10.0)
        if result is not None:
            best_rows = result
            best_max = new_bound
            current_bound = new_bound
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        
        # SA polish attempt
        if time.time() < deadline - 8.0:
            polished, polished_max = sa_polish(best_rows, best_max, deadline - 5.0)
            if polished is not None and polished_max < best_max:
                best_rows = polished
                best_max = polished_max
                current_bound = polished_max
                no_improve_count = 0
    
    # Final polish
    if time.time() < deadline - 3.0:
        polished, polished_max = sa_polish(best_rows, best_max, deadline - 2.0)
        if polished is not None and polished_max < best_max:
            best_rows = polished
            best_max = polished_max
    
    return best_rows
