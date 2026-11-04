"""Difference triangle set solver with improved backtracking and local search."""
import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Bitset-based difference tracking
    # For a row with k+1 elements, the max difference is M, so we need bits 1..M
    # We use Python integers as bitsets
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def compute_row_diffs(row):
        """Compute all pairwise differences for a row as a bitset."""
        diffs = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diffs |= (1 << d)
        return diffs
    
    def validate(rows):
        all_diffs = 0
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            row_diffs = compute_row_diffs(row)
            if row_diffs & all_diffs:
                return False
            all_diffs |= row_diffs
        return True
    
    # Greedy construction with bitset
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
            rows.append(row)
        return rows
    
    # Backtracking with dynamic ordering and bitset
    def backtrack_solve(M, deadline_t, randomize_order=False, max_rows=None):
        """
        Try to build n_rows rows of k+1 elements each, all <= M,
        with all pairwise differences distinct across all rows.
        Returns list of rows or None.
        """
        if max_rows is None:
            max_rows = n_rows
        
        # global_used: bitset of all differences used so far
        # We build rows one at a time
        
        def solve_row(idx, global_used):
            if idx == max_rows:
                return []
            
            def build(pos, vals, row_used):
                if time.time() > deadline_t:
                    raise TimeoutError()
                
                if pos == k + 1:
                    # Row complete, check global conflict
                    if row_used & global_used:
                        return None
                    res = solve_row(idx + 1, global_used | row_used)
                    if res is None:
                        return None
                    return [vals] + res
                
                last = vals[-1]
                
                # Generate candidates: values from last+1 to M
                # Use heuristic: prefer values that create fewer conflicts
                candidates = []
                for v in range(last + 1, M + 1):
                    # Quick check: can we even fit remaining elements?
                    # Need (k+1 - pos) more elements, each at least 1 apart
                    if M - v < (k + 1 - pos) - 1:
                        break
                    
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
                    
                    # Score: prefer values that use larger differences (less likely to conflict later)
                    # and values that are "spread out"
                    score = 0
                    # Count how many new differences are small
                    small_count = 0
                    temp = new_diffs
                    while temp:
                        lsb = temp & (-temp)
                        d = lsb.bit_length() - 1
                        if d < M // 3:
                            small_count += 1
                        temp ^= lsb
                    score = -small_count * 10 + (M - v)  # Prefer larger v, fewer small diffs
                    
                    candidates.append((score, v, new_diffs))
                
                # Sort by score (higher is better)
                candidates.sort(key=lambda x: -x[0])
                
                if randomize_order and candidates:
                    # Add some randomness to top candidates
                    top_n = min(5, len(candidates))
                    top_candidates = candidates[:top_n]
                    random.shuffle(top_candidates)
                    candidates = top_candidates + candidates[top_n:]
                
                for score, v, new_diffs in candidates:
                    vals.append(v)
                    res = build(pos + 1, vals, row_used | new_diffs)
                    if res is not None:
                        return res
                    vals.pop()
                
                return None
            
            return build(1, [0], 0)
        
        return solve_row(0, 0)
    
    # Local search: try to reduce max entry
    def local_search(rows, deadline_t, max_iter=100):
        rows = [list(r) for r in rows]
        
        # Compute current used diffs
        all_diffs = 0
        for row in rows:
            all_diffs |= compute_row_diffs(row)
        
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        
        for iteration in range(max_iter):
            if time.time() > deadline_t:
                break
            
            # Pick a random row to try to improve
            row_idx = random.randint(0, n_rows - 1)
            old_row = list(rows[row_idx])
            old_diffs = compute_row_diffs(old_row)
            
            # Try to find a better row with smaller max
            current_max = old_row[-1]
            improved = False
            
            # Try reducing max by 1, 2, ... up to 10
            for reduction in range(1, min(11, current_max)):
                new_max = current_max - reduction
                if new_max < k:  # Minimum possible max is k
                    break
                
                # Try random constructions with this max
                for attempt in range(20):
                    new_row = [0]
                    valid = True
                    for pos in range(1, k + 1):
                        # Try to pick a value
                        candidates = list(range(new_row[-1] + 1, new_max + 1))
                        random.shuffle(candidates)
                        
                        found = False
                        for v in candidates:
                            # Check if this creates conflicts
                            new_diffs = 0
                            conflict = False
                            for u in new_row:
                                d = v - u
                                if (new_diffs >> d) & 1:
                                    conflict = True
                                    break
                                new_diffs |= (1 << d)
                            
                            if conflict:
                                continue
                            
                            # Check against other rows' diffs (excluding this row)
                            other_diffs = all_diffs & ~old_diffs
                            if new_diffs & other_diffs:
                                continue
                            
                            # Check against this row's already-chosen diffs
                            # (already checked above)
                            
                            new_row.append(v)
                            found = True
                            break
                        
                        if not found:
                            valid = False
                            break
                    
                    if valid and len(new_row) == k + 1:
                        new_row_diffs = compute_row_diffs(new_row)
                        
                        # Check no conflicts with other rows
                        other_diffs = all_diffs & ~old_diffs
                        if not (new_row_diffs & other_diffs):
                            # Update
                            rows[row_idx] = new_row
                            all_diffs = (all_diffs & ~old_diffs) | new_row_diffs
                            
                            new_max_entry = get_max_entry(rows)
                            if new_max_entry < best_max:
                                best_max = new_max_entry
                                best_rows = [list(r) for r in rows]
                            improved = True
                            break
                        else:
                            # Rollback
                            rows[row_idx] = old_row
                            all_diffs = (all_diffs & ~new_row_diffs) | old_diffs
                
                if improved:
                    break
            
            if not improved:
                # Try swapping elements between two rows
                if n_rows >= 2:
                    row_idx1, row_idx2 = random.sample(range(n_rows), 2)
                    row1 = list(rows[row_idx1])
                    row2 = list(rows[row_idx2])
                    
                    if len(row1) > 1 and len(row2) > 1:
                        i1 = random.randint(1, len(row1) - 1)
                        i2 = random.randint(1, len(row2) - 1)
                        
                        # Check if swap is valid
                        # Remove old values, add new values
                        old_v1 = row1[i1]
                        old_v2 = row2[i2]
                        
                        # Try the swap
                        row1[i1] = old_v2
                        row2[i2] = old_v1
                        
                        # Check if rows are still increasing
                        if all(row1[i] < row1[i+1] for i in range(len(row1)-1)) and \
                           all(row2[i] < row2[i+1] for i in range(len(row2)-1)):
                            new_diffs1 = compute_row_diffs(row1)
                            new_diffs2 = compute_row_diffs(row2)
                            
                            other_diffs = all_diffs & ~compute_row_diffs(rows[row_idx1]) & ~compute_row_diffs(rows[row_idx2])
                            
                            if not (new_diffs1 & new_diffs2) and not (new_diffs1 & other_diffs) and not (new_diffs2 & other_diffs):
                                # Valid swap
                                rows[row_idx1] = row1
                                rows[row_idx2] = row2
                                all_diffs = (all_diffs & ~compute_row_diffs(rows[row_idx1]) & ~compute_row_diffs(rows[row_idx2])) | new_diffs1 | new_diffs2
                                
                                new_max_entry = get_max_entry(rows)
                                if new_max_entry < best_max:
                                    best_max = new_max_entry
                                    best_rows = [list(r) for r in rows]
                                improved = True
                        
                        if not improved:
                            # Rollback
                            row1[i1] = old_v1
                            row2[i2] = old_v2
                            rows[row_idx1] = row1
                            rows[row_idx2] = row2
        
        return best_rows, best_max
    
    # Main search
    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    search_deadline = deadline - 5.0
    
    # Phase 1: Try backtracking with various M values
    # Start from lower bound and go up
    lower_bound = total_diffs
    
    # Try a range of M values
    m_values = list(range(lower_bound, best_max + 1))
    random.shuffle(m_values)
    
    # Sort to try smaller M first
    m_values.sort()
    
    for M in m_values[:min(20, len(m_values))]:
        if time.time() > search_deadline:
            break
        
        remaining_time = search_deadline - time.time()
        time_slice = min(remaining_time / 15, 5.0)
        time_slice = max(time_slice, 1.0)
        
        deadline_for_this = time.time() + time_slice
        
        try:
            sol = backtrack_solve(M, deadline_for_this, randomize_order=True)
            if sol:
                curr_max = get_max_entry(sol)
                if curr_max < best_max:
                    best_max = curr_max
                    best_rows = sol
                    # Continue trying smaller M
                else:
                    break
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Phase 2: Local search to improve current best
    if time.time() < search_deadline:
        remaining_time = search_deadline - time.time()
        try:
            improved_rows, improved_max = local_search(best_rows, search_deadline, max_iter=50)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows
        except Exception:
            pass
    
    # Phase 3: More randomized restarts if time allows
    if time.time() < deadline - 10:
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
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
    if not validate(best_rows):
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
    
    return best_rows
