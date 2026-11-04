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
    
    # 1. Greedy Baseline
    def greedy_solution():
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

    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    # 2. CSP Backtracking with Forward Checking and Heuristics
    def csp_search(M, deadline_t, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        # Global used differences bitset
        global_used = 0
        rows = []
        
        # We build rows one by one.
        # To improve search, we can randomize the order of rows, but they are symmetric.
        # The main gain comes from the heuristic within the row construction.
        
        def solve_row(idx, global_used):
            if idx == n_rows:
                return []
            
            def build_row(pos, vals, row_used):
                if time.time() > deadline_t:
                    raise TimeoutError()
                
                if pos == k + 1:
                    # Row complete
                    if row_used & global_used:
                        return None
                    # Recurse to next row
                    res = solve_row(idx + 1, global_used | row_used)
                    if res is None:
                        return None
                    return [vals] + res
                
                last = vals[-1]
                # Candidates are last + 1 to M
                # Heuristic: We want to choose values that don't waste "difference space".
                # A simple heuristic is to try smaller values first, but randomization helps.
                # Let's try a mix: mostly increasing, but with some random jumps.
                
                # Feasibility Check:
                # Remaining positions: rem_pos = k + 1 - pos
                # We need to add rem_pos numbers.
                # The new differences will be between the new numbers and all previous numbers.
                # Total new diffs = rem_pos * len(vals) + rem_pos * (rem_pos - 1) // 2
                # These diffs must be distinct and not in (global_used | row_used).
                # Also, they must be <= M.
                
                rem_pos = k + 1 - pos
                needed_diffs = rem_pos * len(vals) + rem_pos * (rem_pos - 1) // 2
                
                # Quick check: Are there enough available diffs in [1, M]?
                # This is a necessary condition, not sufficient.
                # Counting bits is O(M/word_size), which is fast for M~200.
                # available_mask = ((1 << (M + 1)) - 1) & ~(global_used | row_used)
                # if bin(available_mask).count('1') < needed_diffs: return None
                # This check is too weak because the diffs must be *specific* values derived from the row structure.
                
                candidates = list(range(last + 1, M + 1))
                if random_seed is not None:
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
                    
                    # Check global conflict
                    if new_diffs & global_used:
                        continue
                    
                    # Look-Ahead Feasibility:
                    # If we pick v, can we still complete the row?
                    # A strong look-ahead: Check if the *minimum* possible value for the next element
                    # allows enough room.
                    # For now, let's just recurse. The pruning above is the main cost.
                    
                    vals.append(v)
                    res = build_row(pos + 1, vals, row_used | new_diffs)
                    if res is not None:
                        return res
                    vals.pop()
                
                return None
            
            return build_row(1, [0], 0)
        
        return solve_row(0, 0)

    # 3. Main Search Loop
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
                sol = csp_search(M, deadline_for_this, random_seed=seed)
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
