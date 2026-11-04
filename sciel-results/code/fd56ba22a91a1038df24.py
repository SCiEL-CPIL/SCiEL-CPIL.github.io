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
    
    # 1. Greedy Baseline with bitset optimization
    def greedy_solution(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = 0
                conflict = False
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            if randomize:
                # Slightly perturb: swap some elements
                if len(row) > 2 and random.random() < 0.3:
                    i1, i2 = random.sample(range(1, len(row)), 2)
                    row[i1], row[i2] = row[i2], row[i1]
                    row = [0] + sorted(row[1:])
            rows.append(row)
        return rows

    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    # 2. Backtracking with randomized restarts and bitset optimization
    def find_solution_backtrack(M, deadline_t, randomize_order=False, seed=None):
        if seed is not None:
            random.seed(seed)
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
    
    # 3. Local search: try to improve by modifying rows
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
                    
                    # Try swapping the last elements
                    if row1[-1] != row2[-1]:
                        old_max1 = row1[-1]
                        old_max2 = row2[-1]
                        row1[-1], row2[-1] = row2[-1], row1[-1]
                        
                        # Check if both rows are still valid
                        if row1 == sorted(row1) and row2 == sorted(row2):
                            # Recompute diffs
                            diff1 = set()
                            diff2 = set()
                            for i in range(len(row1)):
                                for j in range(i+1, len(row1)):
                                    diff1.add(row1[j] - row1[i])
                            for i in range(len(row2)):
                                for j in range(i+1, len(row2)):
                                    diff2.add(row2[j] - row2[i])
                            
                            if not (diff1 & diff2) and not (diff1 & used_diffs) and not (diff2 & used_diffs):
                                # Valid swap
                                rows[row_idx1] = row1
                                rows[row_idx2] = row2
                                new_max_entry = get_max_entry(rows)
                                if new_max_entry < best_max:
                                    best_max = new_max_entry
                                    best_rows = [list(r) for r in rows]
                                improved = True
                            
                            if not improved:
                                # Rollback
                                row1[-1], row2[-1] = old_max1, old_max2
        
        return best_rows, best_max
    
    # 4. Main Search Loop
    # We try M values starting from lower_bound up to best_max.
    # We use randomized restarts for each M.
    
    candidates_M = list(range(lower_bound, best_max + 1))
    
    # Time budget allocation
    # We will spend most of the time on the smallest M values, as they are the most likely to be the optimum.
    # But we also need to ensure we return a valid solution.
    
    search_deadline = deadline - 5.0
    
    # Try M from lower_bound to best_max - 1
    # If we find a solution for M, that's better than best_max.
    
    for M in candidates_M:
        if time.time() > search_deadline:
            break
            
        # If M is already >= best_max, we don't need to search unless we want to find a *different* solution with same max (not needed for score).
        if M >= best_max:
            break
            
        # Allocate time for this M
        # Smaller M are harder, give them more time?
        # Or give equal time slices?
        # Let's give each M a time slice of ~2 seconds, with random restarts.
        
        time_slice = 2.0
        restarts = 0
        max_restarts = 5
        
        while time.time() < search_deadline and restarts < max_restarts:
            deadline_for_this = time.time() + time_slice
            seed = restarts * 12345 + M
            try:
                sol = find_solution_backtrack(M, deadline_for_this, randomize_order=True, seed=seed)
                if sol:
                    curr_max = get_max_entry(sol)
                    if curr_max < best_max:
                        best_max = curr_max
                        best_rows = sol
                        # If we found a solution, we can try to see if we can go lower?
                        # The loop continues to the next M (which is lower? No, M is ascending).
                        # Wait, candidates_M is ascending.
                        # If we find a solution for M=113, best_max becomes 113.
                        # The next M in loop is 114. But 114 > 113.
                        # So we should break if we found a solution?
                        # No, we are iterating M from lower_bound upwards.
                        # The first M we find a solution for is the optimal (or close to it).
                        # So we should break out of the M loop if we find a solution?
                        # Actually, if we find a solution for M, it's valid.
                        # Since we are trying M in increasing order, the first solution found is the best possible M for this search strategy.
                        # So we can break.
                        break
            except TimeoutError:
                pass
            except Exception:
                pass
            
            restarts += 1
            # Reduce time slice for subsequent restarts if we are running out of time
            if time.time() > search_deadline - 10:
                time_slice = 1.0
        
        if best_max <= M:
            # We found a solution for this M or better
            break
    
    # 5. Local search refinement
    if time.time() < search_deadline:
        ls_deadline = deadline - 2.0
        try:
            refined_rows, refined_max = local_search(best_rows, ls_deadline, max_iter=50)
            if refined_max < best_max:
                best_max = refined_max
                best_rows = refined_rows
        except Exception:
            pass
    
    # Final Validation
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
