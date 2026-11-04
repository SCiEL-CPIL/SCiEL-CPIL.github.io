"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # 1. Greedy Baseline with randomization
    def greedy_solution(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            if randomize:
                # Slightly perturb: swap some elements
                if len(row) > 2 and random.random() < 0.3:
                    i1, i2 = random.sample(range(1, len(row)), 2)
                    row[i1], row[i2] = row[i2], row[i1]
                    # Re-sort but keep 0 at front
                    row = [0] + sorted(row[1:])
            rows.append(row)
        return rows

    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    # 2. Backtracking with randomized restarts
    def find_solution_backtrack(M, deadline_t, randomize_order=False):
        # global_used is an integer bitset
        # We build rows one by one
        
        def solve_row(idx, global_used):
            if idx == n_rows:
                return []
            
            # Build the current row
            def build(pos, vals, row_used):
                if time.time() > deadline_t:
                    raise TimeoutError()
                
                if pos == k + 1:
                    # Row complete
                    # Check global conflict
                    if row_used & global_used:
                        return None
                    # Recurse to next row
                    res = solve_row(idx + 1, global_used | row_used)
                    if res is None:
                        return None
                    return [vals] + res
                
                last = vals[-1]
                # Generate candidate values
                candidates = list(range(last + 1, M + 1))
                if randomize_order:
                    random.shuffle(candidates)
                
                for v in candidates:
                    # Check internal conflicts
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
                    
                    # Check global conflict (pre-check)
                    if new_diffs & global_used:
                        continue
                    
                    vals.append(v)
                    res = build(pos + 1, vals, row_used | new_diffs)
                    if res is not None:
                        return res
                    vals.pop()
                
                return None
            
            # Start building row with 0
            return build(1, [0], 0)
        
        return solve_row(0, 0)

    # 3. CSP-based search with bitset constraint propagation
    def csp_search(M, deadline_t):
        """
        Constraint Satisfaction Problem approach using bitsets for fast pruning.
        Each row is built with constraint propagation to minimize conflicts.
        """
        global_used = 0  # Bitset of all used differences
        rows = []
        
        # Order rows by difficulty: rows that need more unique differences first
        # For DTS, all rows have same number of diffs, so use random order
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        def build_row(row_idx, current_row, row_used):
            """Build a single row using backtracking with constraint propagation."""
            if len(current_row) == k + 1:
                # Row complete, check global constraint
                if row_used & global_used:
                    return None
                return current_row
            
            last = current_row[-1]
            # Generate candidates in order of least constraint first
            candidates = []
            for v in range(last + 1, M + 1):
                # Check if this value creates any conflicts
                new_diffs = 0
                valid = True
                for u in current_row:
                    d = v - u
                    if (row_used >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                
                if not valid:
                    continue
                
                # Check global constraint
                if new_diffs & global_used:
                    continue
                
                # Heuristic: prefer values that leave more room for future values
                # Count how many future values would be blocked
                candidates.append((v, new_diffs))
            
            # Sort by heuristic: try values that don't block too many future options
            # But for now, just try in order
            for v, new_diffs in candidates:
                if time.time() > deadline_t:
                    raise TimeoutError()
                
                current_row.append(v)
                result = build_row(row_idx, current_row, row_used | new_diffs)
                if result is not None:
                    return result
                current_row.pop()
            
            return None
        
        # Try to build all rows
        def solve_all(idx):
            if idx == n_rows:
                return rows
            
            if time.time() > deadline_t:
                raise TimeoutError()
            
            row_idx = row_order[idx]
            current_row = [0]
            result = build_row(row_idx, current_row, 0)
            
            if result is None:
                return None
            
            # Add to global used
            row_used = 0
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    row_used |= (1 << (result[j] - result[i]))
            
            global_used |= row_used
            rows.append(result)
            
            res = solve_all(idx + 1)
            
            if res is not None:
                return res
            
            # Backtrack
            global_used ^= row_used
            rows.pop()
            return None
        
        return solve_all(0)

    # 4. Local search: try to improve by modifying rows
    def local_search(rows, deadline_t, max_iter=100):
        rows = [list(r) for r in rows]
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        
        for iteration in range(max_iter):
            if time.time() > deadline_t:
                break
            
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            row = list(rows[row_idx])
            
            # Try to reduce the max entry of this row
            current_max = row[-1]
            improved = False
            
            for new_max in range(max(2, current_max - 1), 0, -1):
                # Try to find a valid row ending at or before new_max
                # Simple approach: try random perturbations
                for attempt in range(10):
                    new_row = [0]
                    for pos in range(1, k + 1):
                        # Try values from 1 to new_max
                        candidates = list(range(1, new_max + 1))
                        random.shuffle(candidates)
                        found = False
                        for v in candidates:
                            if v <= new_row[-1]:
                                continue
                            # Check conflicts
                            new_diffs = {v - a for a in new_row}
                            if new_diffs & used_diffs:
                                continue
                            # Check internal conflicts
                            if len(new_diffs) != len(new_row):
                                continue
                            new_row.append(v)
                            found = True
                            break
                        if not found:
                            break
                    
                    if len(new_row) == k + 1 and new_row[-1] <= new_max:
                        # Check if this row's diffs conflict with other rows
                        row_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i+1, len(new_row)):
                                row_diffs.add(new_row[j] - new_row[i])
                        
                        # Remove old row's diffs
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                used_diffs.discard(row[j] - row[i])
                        
                        # Check no conflicts
                        if not (row_diffs & used_diffs):
                            # Add new diffs
                            used_diffs |= row_diffs
                            rows[row_idx] = new_row
                            new_max_entry = get_max_entry(rows)
                            if new_max_entry < best_max:
                                best_max = new_max_entry
                                best_rows = [list(r) for r in rows]
                            improved = True
                            break
                        
                        # Rollback
                        used_diffs -= row_diffs
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                used_diffs.add(row[j] - row[i])
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try a different strategy: swap elements within a row
                if n_rows > 1:
                    row_idx1, row_idx2 = random.sample(range(n_rows), 2)
                    row1 = list(rows[row_idx1])
                    row2 = list(rows[row_idx2])
                    
                    # Try swapping some elements
                    if len(row1) > 1 and len(row2) > 1:
                        i1 = random.randint(1, len(row1) - 1)
                        i2 = random.randint(1, len(row2) - 1)
                        
                        # Remove diffs from both rows
                        diffs1 = set()
                        for i in range(len(row1)):
                            for j in range(i+1, len(row1)):
                                diffs1.add(row1[j] - row1[i])
                        diffs2 = set()
                        for i in range(len(row2)):
                            for j in range(i+1, len(row2)):
                                diffs2.add(row2[j] - row2[i])
                        
                        used_diffs -= diffs1
                        used_diffs -= diffs2
                        
                        # Swap
                        row1[i1], row2[i2] = row2[i2], row1[i1]
                        
                        # Check validity
                        valid = True
                        new_diffs1 = set()
                        for i in range(len(row1)):
                            for j in range(i+1, len(row1)):
                                d = row1[j] - row1[i]
                                if d in new_diffs1:
                                    valid = False
                                    break
                                new_diffs1.add(d)
                        if valid:
                            new_diffs2 = set()
                            for i in range(len(row2)):
                                for j in range(i+1, len(row2)):
                                    d = row2[j] - row2[i]
                                    if d in new_diffs2:
                                        valid = False
                                        break
                                    new_diffs2.add(d)
                        
                        if valid and not (new_diffs1 & new_diffs2) and not (new_diffs1 & used_diffs) and not (new_diffs2 & used_diffs):
                            # Check rows are still increasing
                            if all(row1[i] < row1[i+1] for i in range(len(row1)-1)) and \
                               all(row2[i] < row2[i+1] for i in range(len(row2)-1)):
                                rows[row_idx1] = row1
                                rows[row_idx2] = row2
                                used_diffs |= new_diffs1
                                used_diffs |= new_diffs2
                                new_max_entry = get_max_entry(rows)
                                if new_max_entry < best_max:
                                    best_max = new_max_entry
                                    best_rows = [list(r) for r in rows]
                                improved = True
                        
                        if not improved:
                            # Rollback
                            used_diffs |= diffs1
                            used_diffs |= diffs2
                            rows[row_idx1] = list(row1)
                            rows[row_idx2] = list(row2)
        
        return best_rows, best_max
    
    # 5. Main search loop with multiple strategies
    search_deadline = deadline - 5.0  # Leave 5s for final validation
    
    # Phase 1: Try CSP-based search with various M values
    # Start from a reasonable upper bound and work down
    M_start = best_max - 1
    M_end = max(lower_bound, M_start - 20)
    
    for M in range(M_start, M_end - 1, -1):
        if time.time() > search_deadline:
            break
        
        remaining_time = search_deadline - time.time()
        time_slice = min(remaining_time / 15, 2.0)
        time_slice = max(time_slice, 0.3)
        
        deadline_for_this = time.time() + time_slice
        
        try:
            sol = csp_search(M, deadline_for_this)
            if sol:
                curr_max = get_max_entry(sol)
                if curr_max < best_max:
                    best_max = curr_max
                    best_rows = sol
                    # Don't break - try to find even better
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Phase 2: Local search to improve current best
    if time.time() < search_deadline:
        try:
            improved_rows, improved_max = local_search(best_rows, search_deadline, max_iter=50)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        except Exception:
            pass
    
    # Phase 3: More randomized restarts if time allows
    if time.time() < deadline - 10:
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Try different random seeds
            seed = restart * 1000 + 42
            greedy_rows = greedy_solution(randomize=True, seed=seed)
            greedy_max = get_max_entry(greedy_rows)
            
            if greedy_max < best_max:
                best_max = greedy_max
                best_rows = greedy_rows
            
            # Try local search on this solution
            if time.time() < deadline - 5:
                try:
                    improved_rows, improved_max = local_search(greedy_rows, deadline - 5, max_iter=20)
                    if improved_max < best_max:
                        best_max = improved_max
                        best_rows = improved_rows
                except Exception:
                    pass
    
    # Final validation
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
    
    if not validate(best_rows):
        # Fallback to pure greedy
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
    
    return best_rows

# EVOLVE-BLOCK-END
