import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Build a solid initial solution using a fast greedy algorithm.
    def greedy_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_initial()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # Helper to compute max of rows
    def get_max(rows):
        return max(max(r) for r in rows)

    # Helper to compute diffs of a row
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    # 2. Improved rebuild with difference-pressure candidate selection
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, pressure_weight=1.0):
        """
        Builds a row using a hybrid strategy:
        - Collect valid candidates
        - Score each candidate by a combination of its value and the max new difference it introduces
        - Prefer candidates with low score (small value, low max new diff)
        """
        row = [0]
        local_used = set()
        
        for step in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates up to max_val_limit
            # Limit search to avoid excessive time; we'll collect up to a reasonable number
            max_candidates_to_check = 50
            checked = 0
            
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    # Score: combination of x and max new diff
                    max_new_diff = max(new_diffs)
                    score = x * (1.0 - pressure_weight) + max_new_diff * pressure_weight
                    valid_candidates.append((x, new_diffs, score))
                    checked += 1
                    if checked >= max_candidates_to_check:
                        break
            
            if not valid_candidates:
                return None
            
            # Sort by score and pick from top few with some randomness
            valid_candidates.sort(key=lambda t: t[2])
            
            # Pick from top min(5, len) candidates with weighted probability favoring lower scores
            n_top = min(5, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]
            
            # Weighted random selection: lower score = higher probability
            weights = [1.0 / (1.0 + c[2] * 0.1) for c in top_candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = top_candidates[0]
            for c, w in zip(top_candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c
                    break
            
            x, new_diffs, _ = chosen
            row.append(x)
            local_used.update(new_diffs)
            
            if x > max_val_limit:
                return None
                
        return row

    # 3. Multi-row rebuild: try to rebuild 2 rows simultaneously
    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit, row_i, row_j):
        """
        Try to rebuild rows i and j simultaneously.
        Returns (new_row_i, new_row_j) or (None, None) if failed.
        """
        # First, build row i with forbidden_diffs
        row_i_new = rebuild_row(k, forbidden_diffs, rng, max_val_limit)
        if row_i_new is None:
            return None, None
        
        diffs_i = get_diffs(row_i_new)
        
        # Now build row j with forbidden_diffs + diffs_i
        forbidden_j = forbidden_diffs | diffs_i
        row_j_new = rebuild_row(k, forbidden_j, rng, max_val_limit)
        if row_j_new is None:
            return None, None
        
        return row_i_new, row_j_new

    # Main Loop with simulated annealing
    rng = random.Random(12345)
    
    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.995
    min_temp = 0.01
    temp = initial_temp
    
    no_improve_count = 0
    total_moves = 0
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # Check if we should switch to polish mode (last 15 seconds)
        polish_mode = remaining < 15.0
        
        if polish_mode:
            # In polish mode, only accept improvements and use tighter limits
            target_max = current_max - 1
            accept_neutral = False
            max_candidates = 30
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.15 if temp > 0.1 else 0.05)
            max_candidates = 50

        # Decide whether to do single or multi-row rebuild
        use_multi = (not polish_mode) and (rng.random() < 0.2) and (n_rows >= 2)
        
        if use_multi:
            # Pick two distinct rows
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            if i == j:
                j = (j + 1) % n_rows
            
            old_row_i = rows[i]
            old_row_j = rows[j]
            old_diffs_i = get_diffs(old_row_i)
            old_diffs_j = get_diffs(old_row_j)
            
            # Remove both from used_diffs
            used_diffs.difference_update(old_diffs_i)
            used_diffs.difference_update(old_diffs_j)
            
            # Try to rebuild both
            new_row_i, new_row_j = rebuild_two_rows(
                k, used_diffs, rng, target_max if target_max > k else current_max, i, j
            )
            
            if new_row_i is None or new_row_j is None:
                # Failed, restore
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs_i = get_diffs(new_row_i)
            new_diffs_j = get_diffs(new_row_j)
            
            # Verify no conflicts
            if (new_diffs_i & used_diffs) or (new_diffs_j & used_diffs) or (new_diffs_i & new_diffs_j):
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            # Calculate new global max
            rows[i] = new_row_i
            rows[j] = new_row_j
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                used_diffs.update(new_diffs_i)
                used_diffs.update(new_diffs_j)
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)  # Reheat slightly
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs_i)
                    used_diffs.update(new_diffs_j)
                    no_improve_count += 1
                else:
                    rows[i] = old_row_i
                    rows[j] = old_row_j
                    used_diffs.update(old_diffs_i)
                    used_diffs.update(old_diffs_j)
                    no_improve_count += 1
                
                temp *= cooling_rate
                total_moves += 1
        else:
            # Single row rebuild
            i = rng.randint(0, n_rows - 1)
            
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            # Remove from used_diffs
            used_diffs.difference_update(old_diffs)
            
            # Try to rebuild
            new_row = None
            target = target_max if target_max > k else current_max
            new_row = rebuild_row(k, used_diffs, rng, target)
            
            if new_row is None:
                # Try with current_max as fallback
                new_row = rebuild_row(k, used_diffs, rng, current_max)
            
            if new_row is None:
                used_diffs.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs = get_diffs(new_row)
            
            # Verify no conflicts
            if new_diffs & used_diffs:
                used_diffs.update(old_diffs)
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            rows[i] = new_row
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                used_diffs.update(new_diffs)
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs)
                    no_improve_count += 1
                else:
                    rows[i] = old_row
                    used_diffs.update(old_diffs)
                    no_improve_count += 1
                
                temp *= cooling_rate
                total_moves += 1
        
        # If we are stuck for a long time, try a bigger jump
        if no_improve_count > 200 and not polish_mode:
            # Try rebuilding a random subset of rows
            num_to_rebuild = min(n_rows, rng.randint(2, 4))
            indices = rng.sample(range(n_rows), num_to_rebuild)
            
            old_rows = {}
            old_diffs_all = set()
            for idx in indices:
                old_rows[idx] = rows[idx]
                old_diffs_all.update(get_diffs(rows[idx]))
            
            used_diffs.difference_update(old_diffs_all)
            
            success = True
            new_rows = {}
            new_diffs_all = set()
            
            for idx in indices:
                target = target_max if target_max > k else current_max
                new_row = rebuild_row(k, used_diffs | new_diffs_all, rng, target)
                if new_row is None:
                    success = False
                    break
                new_rows[idx] = new_row
                new_diffs_all.update(get_diffs(new_row))
            
            if success and not (new_diffs_all & used_diffs):
                for idx in indices:
                    rows[idx] = new_rows[idx]
                used_diffs.update(new_diffs_all)
                new_global_max = get_max(rows)
                if new_global_max < current_max:
                    current_max = new_global_max
                    best_rows = [r[:] for r in rows]
                    no_improve_count = 0
                    temp = max(temp * 1.2, min_temp)
                else:
                    no_improve_count = 100  # Partial reset
            else:
                # Restore
                for idx in indices:
                    rows[idx] = old_rows[idx]
                used_diffs.update(old_diffs_all)
                no_improve_count = 150
            
            temp *= cooling_rate
            total_moves += 1

        # Check lower bound
        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    return best_rows

# EVOLVE-BLOCK-END
