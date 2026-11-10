import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 140.0
    
    # Helper to calculate differences set for a row
    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    # Helper to calculate bitmask for a row's differences
    def row_mask(row, max_diff):
        mask = 0
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - a
                if d <= max_diff:
                    mask |= (1 << d)
        return mask

    # Greedy baseline construction
    def greedy_row(used_set, k, cap=None, rng=None):
        if rng is None:
            rng = random
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    # Generate baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        if row is None:
            # Fallback if greedy fails (unlikely for small k)
            row = [0] * (k+1)
        rows.append(row)
        used |= row_diffs(row)
    
    baseline_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = baseline_max

    # Strategy 1: Precompute and Swap
    # We try to generate a pool of valid rows up to a certain limit.
    # If the pool is large enough, we can do fast local search.
    # Limit for precomputation: We don't want to exceed baseline_max by much, 
    # but we need enough candidates to find a good swap.
    # Let's try to generate rows up to baseline_max.
    
    def generate_valid_rows(limit, k, time_limit):
        """
        Generates all valid rows of length k+1 starting at 0, with max element <= limit.
        Returns list of (max_val, mask, row_tuple) or None if timeout.
        """
        start = time.time()
        timeout = time_limit
        rows_list = []
        
        # Precompute bits for speed
        # max possible diff is 'limit'
        bit_of = [1 << d for d in range(limit + 1)]
        
        def backtrack(current_row, current_mask, last_val):
            if time.time() - start > timeout:
                return False # Signal timeout
            
            if len(current_row) == k + 1:
                max_val = current_row[-1]
                # Store as tuple for immutability and hashability if needed, but list is fine
                rows_list.append((max_val, current_mask, tuple(current_row)))
                return True
            
            for x in range(last_val + 1, limit + 1):
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if d > limit:
                        valid = False
                        break
                    bit = bit_of[d]
                    if current_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if valid:
                    current_row.append(x)
                    if not backtrack(current_row, current_mask | new_mask, x):
                        return False
                    current_row.pop()
        
        if backtrack([0], 0, 0):
            return rows_list
        else:
            return None

    # Try to generate pool. 
    # For n=7, k=5, baseline is around 110-120. 
    # Generating all rows up to 120 might take a few seconds.
    # We allocate 20s for this phase.
    pool = None
    if time.time() < deadline - 120:
        pool = generate_valid_rows(baseline_max, k, 20.0)
        
        if pool is not None:
            # Sort pool by max_val to allow early termination in search
            pool.sort(key=lambda x: x[0])
            
            # Initialize state with baseline
            current_rows = [list(r) for r in best_rows]
            current_masks = []
            for r in current_rows:
                current_masks.append(row_mask(r, baseline_max))
            
            # Calculate total mask
            total_mask = 0
            for m in current_masks:
                total_mask |= m
                
            # Local Search: "Generate and Swap"
            # We want to minimize the global max.
            # Strategy: Find the row with the largest max. Try to replace it with a smaller one from pool.
            improved = True
            no_improve_count = 0
            rng = random.Random(42)
            
            while improved and time.time() < deadline - 10:
                improved = False
                no_improve_count += 1
                if no_improve_count > 100:
                    break # Prevent infinite loops in case of oscillation
                
                # Find index of row with max value
                max_idx = -1
                max_val = -1
                for i, r in enumerate(current_rows):
                    if r[-1] > max_val:
                        max_val = r[-1]
                        max_idx = i
                
                if max_idx == -1:
                    break
                
                # Calculate mask of all other rows
                others_mask = total_mask ^ current_masks[max_idx]
                
                # Search pool for a candidate
                # We want a candidate with max_val < current max_val
                # And (cand_mask & others_mask) == 0
                found_idx = -1
                best_cand = None
                best_cand_mask = 0
                best_cand_row = None
                
                # Since pool is sorted by max_val, we can iterate until we find one that fits
                # We prefer the smallest possible max_val to minimize the new global max
                for idx, (cand_max, cand_mask, cand_row) in enumerate(pool):
                    if cand_max >= max_val:
                        break # No more smaller candidates
                    
                    if (cand_mask & others_mask) == 0:
                        # Found a valid swap
                        found_idx = idx
                        best_cand = (cand_max, cand_mask, cand_row)
                        break # Take the first (smallest) one found
                
                if best_cand is not None:
                    # Perform swap
                    cand_max, cand_mask, cand_row = best_cand
                    
                    # Update total mask
                    total_mask = (total_mask ^ current_masks[max_idx]) | cand_mask
                    
                    # Update row and mask
                    current_rows[max_idx] = list(cand_row)
                    current_masks[max_idx] = cand_mask
                    
                    improved = True
                    no_improve_count = 0
                    
                    # Update best if global max decreased
                    current_global_max = max(r[-1] for r in current_rows)
                    if current_global_max < best_max:
                        best_max = current_global_max
                        best_rows = [list(r) for r in current_rows]
                        
            # If local search finished, check if we found something better
            if best_max < baseline_max:
                return best_rows

    # Strategy 2: Simulated Annealing / DFS Refinement
    # If precomputation didn't work or wasn't fast enough, fall back to SA.
    # This is the logic from the original program, slightly optimized.
    
    def dfs_row(row, used_set, cap, row_deadline, randomize=False, rng=None):
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None

        candidates = list(range(last + 1, upper + 1))
        if randomize and rng is not None:
            rng.shuffle(candidates)

        for x in candidates:
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline, randomize, rng)
            if res is not None:
                return res
        return None

    # Initialize SA state
    used = set()
    rows = [list(r) for r in best_rows] # Start from best found so far
    for r in rows:
        used |= row_diffs(r)
    
    current_max = max(r[-1] for r in rows)
    
    T = 10.0
    cooling = 0.998
    rng = random.Random(123)
    
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        if T < 0.1:
            break
            
        i = rng.randint(0, n_rows - 1)
        
        # Try to find a row with max <= current_max (or slightly higher)
        # We want to reduce the global max, so we constrain the new row to be <= current_max
        # Ideally < current_max if this row is the bottleneck, but SA allows exploration.
        
        used_without_i = set(used)
        used_without_i.difference_update(row_diffs(rows[i]))
        
        # Set a tight deadline for this single row search
        row_deadline = min(deadline - 5, time.time() + 0.5)
        
        # Try to find a row that fits within current_max
        # Sometimes allow current_max + 1 to escape local minima, but we prefer reduction
        cap = current_max
        
        cand = dfs_row([0], used_without_i, cap, row_deadline, randomize=True, rng=rng)
        
        if cand is not None:
            new_last = cand[-1]
            
            # Calculate new global max
            max_others = 0
            for j in range(n_rows):
                if j != i:
                    if rows[j][-1] > max_others:
                        max_others = rows[j][-1]
            
            new_global_max = max(max_others, new_last)
            delta = new_global_max - current_max
            
            if delta < 0:
                accept = True
            else:
                try:
                    prob = math.exp(-delta / T)
                except:
                    prob = 0.0
                accept = rng.random() < prob
            
            if accept:
                used.difference_update(row_diffs(rows[i]))
                rows[i] = cand
                used |= row_diffs(cand)
                
                current_max = new_global_max
                
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count > 100:
                    # Random restart of a few rows
                    for _ in range(rng.randint(1, 2)):
                        j = rng.randint(0, n_rows - 1)
                        used.difference_update(row_diffs(rows[j]))
                        new_row = dfs_row([0], used, current_max, min(deadline-5, time.time()+0.5), randomize=True, rng=rng)
                        if new_row is not None:
                            rows[j] = new_row
                            used |= row_diffs(new_row)
                            current_max = max(r[-1] for r in rows)
                    no_improve_count = 0
                    
        T *= cooling

    # Final Validation
    def validate_solution(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    return False
            diffs = row_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True

    if not validate_solution(best_rows):
        # Fallback to greedy if something went wrong
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(set(used), k)
            if row is None:
                row = [0] * (k+1)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return best_rows
