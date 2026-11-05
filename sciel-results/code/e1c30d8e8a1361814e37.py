import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def row_diffs_mask(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                mask |= (1 << (row[j] - row[i]))
        return mask

    def all_diffs_mask(rows):
        mask = 0
        for row in rows:
            mask |= row_diffs_mask(row)
        return mask

    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    # Phase 1: Diverse randomized greedy to get good starting points
    def greedy_randomized(seed=None, lookahead=0):
        if seed is not None:
            random.seed(seed)
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                # Generate candidates from last+1 upward, with some randomization
                candidates = []
                for v in range(last + 1, last + 200):
                    candidates.append(v)
                if lookahead > 0 and len(candidates) > lookahead:
                    random.shuffle(candidates)
                    candidates = candidates[:lookahead + 5]
                
                found = False
                for v in candidates:
                    new_mask = 0
                    ok = True
                    for a in row:
                        d = v - a
                        if (used_mask >> d) & 1:
                            ok = False
                            break
                        new_mask |= (1 << d)
                    if ok:
                        # Check internal consistency
                        internal_ok = True
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                if (new_mask >> (row[j] - row[i])) & 1:
                                    # Already in row, that's fine, it's the same diff
                                    pass
                        row.append(v)
                        used_mask |= new_mask
                        found = True
                        break
                if not found:
                    # Try wider search
                    for v in range(last + 1, last + 500):
                        new_mask = 0
                        ok = True
                        for a in row:
                            d = v - a
                            if (used_mask >> d) & 1:
                                ok = False
                                break
                            new_mask |= (1 << d)
                        if ok:
                            row.append(v)
                            used_mask |= new_mask
                            found = True
                            break
                    if not found:
                        # Last resort: very wide
                        for v in range(last + 1, last + 2000):
                            new_mask = 0
                            ok = True
                            for a in row:
                                d = v - a
                                if (used_mask >> d) & 1:
                                    ok = False
                                    break
                                new_mask |= (1 << d)
                            if ok:
                                row.append(v)
                                used_mask |= new_mask
                                found = True
                                break
        return rows

    # Phase 2: CSP backtracking with forward checking and tight M
    def csp_solve(M, deadline_t, seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Precompute: for each row position, what values are feasible given global used
        used_mask = 0
        rows = []
        
        def solve_row(row_idx, global_used):
            if row_idx == n_rows:
                return True
            
            def build(pos, vals, row_used):
                if time.time() > deadline_t:
                    raise TimeoutError()
                
                if pos == k + 1:
                    if row_used & global_used:
                        return False
                    rows.append(list(vals))
                    if solve_row(row_idx + 1, global_used | row_used):
                        return True
                    rows.pop()
                    return False
                
                last = vals[-1]
                # Forward checking: only try values that don't immediately conflict
                candidates = []
                for v in range(last + 1, M + 1):
                    new_diffs = 0
                    conflict = False
                    for u in vals:
                        d = v - u
                        if (row_used >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    if conflict:
                        continue
                    if new_diffs & global_used:
                        continue
                    candidates.append((v, new_diffs))
                
                if not candidates:
                    return False
                
                # Order by smallest value first (or randomize)
                if random.random() < 0.3:
                    random.shuffle(candidates)
                
                for v, new_diffs in candidates:
                    vals.append(v)
                    if build(pos + 1, vals, row_used | new_diffs):
                        return True
                    vals.pop()
                
                return False
            
            return build(1, [0], 0)
        
        return solve_row(0, 0)

    # Phase 3: Simulated annealing to minimize scope
    def simulated_annealing(initial_rows, deadline_t):
        rows = [list(r) for r in initial_rows]
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        
        current_max = best_max
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        
        used_mask = all_diffs_mask(rows)
        
        for iteration in range(20000):
            if time.time() > deadline_t:
                break
            
            if T < T_min:
                break
            
            # Pick a random row and try to modify it
            row_idx = random.randint(0, n_rows - 1)
            row = list(rows[row_idx])
            
            # Compute diffs for this row
            old_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    old_diffs.add(row[j] - row[i])
            
            # Try to rebuild this row with a smaller max
            target_max = row[-1] - random.randint(1, max(1, row[-1] // 5))
            if target_max < 1:
                target_max = 1
            
            new_row = [0]
            for pos in range(1, k + 1):
                candidates = list(range(1, target_max + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    if v <= new_row[-1]:
                        continue
                    new_diffs = set()
                    ok = True
                    for a in new_row:
                        d = v - a
                        if d in new_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    # Check against other rows' diffs
                    other_diffs = set()
                    for r_idx, r in enumerate(rows):
                        if r_idx == row_idx:
                            continue
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                other_diffs.add(r[j] - r[i])
                    if new_diffs & other_diffs:
                        continue
                    new_row.append(v)
                    found = True
                    break
                if not found:
                    break
            
            if len(new_row) == k + 1 and new_row[-1] < current_max:
                # Check full validity
                test_rows = [list(r) if i != row_idx else new_row for i, r in enumerate(rows)]
                if validate(test_rows):
                    rows[row_idx] = new_row
                    current_max = get_max_entry(rows)
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows]
            elif len(new_row) == k + 1:
                # Accept with probability exp(-delta/T)
                new_max = get_max_entry([list(r) if i != row_idx else new_row for i, r in enumerate(rows)])
                delta = new_max - current_max
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    # Check validity
                    test_rows = [list(r) if i != row_idx else new_row for i, r in enumerate(rows)]
                    if validate(test_rows):
                        rows[row_idx] = new_row
                        current_max = new_max
                        if current_max < best_max:
                            best_max = current_max
                            best_rows = [list(r) for r in rows]
            
            T *= cooling
        
        return best_rows, best_max

    # Main search strategy
    best_rows = None
    best_max = float('inf')
    
    # Generate diverse initial solutions
    num_initial = 5
    for i in range(num_initial):
        if time.time() > deadline - 30:
            break
        seed = i * 7919 + 42
        rows = greedy_randomized(seed=seed, lookahead=10 + i * 5)
        curr_max = get_max_entry(rows)
        if curr_max < best_max:
            best_max = curr_max
            best_rows = rows
    
    if best_rows is None:
        best_rows = greedy_randomized(seed=42)
        best_max = get_max_entry(best_rows)
    
    # Try CSP with decreasing M
    search_deadline = deadline - 15
    M_start = best_max - 1
    M_min = max(112, total_diffs)
    
    for M in range(M_start, max(M_min - 1, 100), -1):
        if time.time() > search_deadline:
            break
        if M < best_max - 1:
            continue
        
        time_slice = min((search_deadline - time.time()) / 10, 8.0)
        time_slice = max(time_slice, 1.0)
        dl = time.time() + time_slice
        
        for attempt in range(3):
            if time.time() > search_deadline:
                break
            try:
                sol = csp_solve(M, dl, seed=attempt * 1009 + M)
                if sol:
                    curr_max = get_max_entry(rows)
                    if curr_max < best_max:
                        best_max = curr_max
                        best_rows = [list(r) for r in rows]
                        break
            except TimeoutError:
                pass
            except Exception:
                pass
    
    # Simulated annealing refinement
    if time.time() < deadline - 10:
        try:
            improved_rows, improved_max = simulated_annealing(best_rows, deadline - 5)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        except Exception:
            pass
    
    # Final validation
    if not validate(best_rows):
        best_rows = greedy_randomized(seed=12345)
        best_max = get_max_entry(best_rows)
    
    return best_rows
