import random
import time
import math
from collections import Counter

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

    # 2. Rebuild row with UCB-style gap selection
    def rebuild_row_ucb(k, forbidden_diffs, rng, max_val_limit, diff_histogram=None):
        """
        Builds a row using UCB-style candidate selection.
        Each candidate gap is scored by UCB1 balancing immediate compactness
        and long-term difference collision avoidance.
        """
        row = [0]
        local_used = set()
        
        # Initialize histogram if not provided
        if diff_histogram is None:
            diff_histogram = Counter()
        
        for step in range(k):
            last = row[-1]
            valid_candidates = []
            
            # Search for valid candidates up to max_val_limit
            max_candidates_to_check = 80
            checked = 0
            
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    # Compute UCB score for this candidate
                    # Immediate reward: negative of x (prefer small)
                    # Long-term: collision risk based on histogram
                    max_new_diff = max(new_diffs)
                    
                    # Estimate collision risk
                    collision_risk = 0.0
                    for d in new_diffs:
                        # Check nearby differences in histogram
                        for offset in range(-2, 3):
                            collision_risk += diff_histogram.get(d + offset, 0)
                    
                    # UCB score: prefer low x and low collision risk
                    # Higher score = better candidate
                    score = -x * 0.1 - collision_risk * 0.5 - max_new_diff * 0.05
                    
                    valid_candidates.append((x, new_diffs, score))
                    checked += 1
                    if checked >= max_candidates_to_check:
                        break
            
            if not valid_candidates:
                return None
            
            # Sort by score (descending) and pick from top few
            valid_candidates.sort(key=lambda t: t[2], reverse=True)
            
            # Pick from top min(8, len) candidates with weighted probability
            n_top = min(8, len(valid_candidates))
            top_candidates = valid_candidates[:n_top]
            
            # Weighted random selection: higher score = higher probability
            # Use softmax-like weighting
            max_score = top_candidates[0][2]
            weights = [math.exp((c[2] - max_score) / 0.5) for c in top_candidates]
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
            
            # Update histogram with new differences
            for d in new_diffs:
                diff_histogram[d] += 1
            
            if x > max_val_limit:
                return None
                
        return row

    # 3. Multi-row rebuild: try to rebuild 2 rows simultaneously
    def rebuild_two_rows(k, forbidden_diffs, rng, max_val_limit, diff_histogram):
        """
        Try to rebuild two rows simultaneously.
        Returns (new_row_i, new_row_j) or (None, None) if failed.
        """
        # First, build row i with forbidden_diffs
        row_i_new = rebuild_row_ucb(k, forbidden_diffs, rng, max_val_limit, diff_histogram)
        if row_i_new is None:
            return None, None
        
        diffs_i = get_diffs(row_i_new)
        
        # Now build row j with forbidden_diffs + diffs_i
        forbidden_j = forbidden_diffs | diffs_i
        row_j_new = rebuild_row_ucb(k, forbidden_j, rng, max_val_limit, diff_histogram)
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
    
    # Build global difference histogram for UCB scoring
    global_diff_histogram = Counter()
    for row in rows:
        for d in get_diffs(row):
            global_diff_histogram[d] += 1

    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        # Check if we should switch to polish mode (last 15 seconds)
        polish_mode = remaining < 15.0
        
        if polish_mode:
            # In polish mode, only accept improvements and use tighter limits
            target_max = current_max - 1
            accept_neutral = False
            max_candidates = 50
        else:
            target_max = current_max - 1
            accept_neutral = rng.random() < (0.15 if temp > 0.1 else 0.05)
            max_candidates = 80

        # Decide whether to do single or multi-row rebuild
        use_multi = (not polish_mode) and (rng.random() < 0.25) and (n_rows >= 2)
        
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
            
            # Remove from histogram
            for d in old_diffs_i:
                global_diff_histogram[d] = max(0, global_diff_histogram[d] - 1)
            for d in old_diffs_j:
                global_diff_histogram[d] = max(0, global_diff_histogram[d] - 1)
            
            # Try to rebuild both
            target = target_max if target_max > k else current_max
            new_row_i, new_row_j = rebuild_two_rows(
                k, used_diffs, rng, target, global_diff_histogram
            )
            
            if new_row_i is None or new_row_j is None:
                # Failed, restore
                used_diffs.update(old_diffs_i)
                used_diffs.update(old_diffs_j)
                for d in old_diffs_i:
                    global_diff_histogram[d] += 1
                for d in old_diffs_j:
                    global_diff_histogram[d] += 1
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
                for d in old_diffs_i:
                    global_diff_histogram[d] += 1
                for d in old_diffs_j:
                    global_diff_histogram[d] += 1
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
                for d in new_diffs_i:
                    global_diff_histogram[d] += 1
                for d in new_diffs_j:
                    global_diff_histogram[d] += 1
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)  # Reheat slightly
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs_i)
                    used_diffs.update(new_diffs_j)
                    for d in new_diffs_i:
                        global_diff_histogram[d] += 1
                    for d in new_diffs_j:
                        global_diff_histogram[d] += 1
                else:
                    # Revert
                    rows[i] = old_row_i
                    rows[j] = old_row_j
                    used_diffs.update(old_diffs_i)
                    used_diffs.update(old_diffs_j)
                    for d in old_diffs_i:
                        global_diff_histogram[d] += 1
                    for d in old_diffs_j:
                        global_diff_histogram[d] += 1
                    no_improve_count += 1
            
            temp *= cooling_rate
            total_moves += 1
        else:
            # Single row rebuild
            i = rng.randint(0, n_rows - 1)
            
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            used_diffs.difference_update(old_diffs)
            
            # Remove from histogram
            for d in old_diffs:
                global_diff_histogram[d] = max(0, global_diff_histogram[d] - 1)
            
            new_row = None
            target = target_max if target_max > k else current_max
            
            # Try UCB rebuild at reduced max
            new_row = rebuild_row_ucb(k, used_diffs, rng, target, global_diff_histogram)
            
            if new_row is None:
                # Try at current max
                new_row = rebuild_row_ucb(k, used_diffs, rng, current_max, global_diff_histogram)
            
            if new_row is None:
                used_diffs.update(old_diffs)
                for d in old_diffs:
                    global_diff_histogram[d] += 1
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            new_diffs = get_diffs(new_row)
            
            if new_diffs & used_diffs:
                used_diffs.update(old_diffs)
                for d in old_diffs:
                    global_diff_histogram[d] += 1
                no_improve_count += 1
                temp *= cooling_rate
                total_moves += 1
                continue
            
            rows[i] = new_row
            new_global_max = get_max(rows)
            
            if new_global_max < current_max:
                used_diffs.update(new_diffs)
                for d in new_diffs:
                    global_diff_histogram[d] += 1
                current_max = new_global_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
                temp = max(temp * 1.1, min_temp)  # Reheat slightly
            else:
                if accept_neutral:
                    used_diffs.update(new_diffs)
                    for d in new_diffs:
                        global_diff_histogram[d] += 1
                else:
                    # Revert
                    rows[i] = old_row
                    used_diffs.update(old_diffs)
                    for d in old_diffs:
                        global_diff_histogram[d] += 1
                    no_improve_count += 1
            
            temp *= cooling_rate
            total_moves += 1
            
            # Early termination if we hit the lower bound
            lb = n_rows * k * (k + 1) // 2
            if current_max == lb:
                break

    # Phase 2: Final polish - try to reduce max by 1 more
    polish_deadline = start_time + 118.0
    while time.time() < polish_deadline:
        # Find which row(s) have the max entry
        max_val = current_max
        rows_with_max = [i for i, r in enumerate(rows) if max(r) == max_val]
        
        if not rows_with_max:
            break
        
        # Try to rebuild each row with max to fit in max_val - 1
        improved = False
        for i in rows_with_max:
            if time.time() >= polish_deadline:
                break
            
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            used_diffs.difference_update(old_diffs)
            
            # Remove from histogram
            for d in old_diffs:
                global_diff_histogram[d] = max(0, global_diff_histogram[d] - 1)
            
            target = max_val - 1
            new_row = rebuild_row_ucb(k, used_diffs, rng, target, global_diff_histogram)
            
            if new_row is not None:
                new_diffs = get_diffs(new_row)
                if not (new_diffs & used_diffs):
                    rows[i] = new_row
                    used_diffs.update(new_diffs)
                    for d in new_diffs:
                        global_diff_histogram[d] += 1
                    current_max = max(max(r) for r in rows)
                    best_rows = [r[:] for r in rows]
                    improved = True
                    break
                else:
                    used_diffs.update(old_diffs)
                    for d in old_diffs:
                        global_diff_histogram[d] += 1
            else:
                used_diffs.update(old_diffs)
                for d in old_diffs:
                    global_diff_histogram[d] += 1
        
        if not improved:
            break

    return best_rows

# EVOLVE-BLOCK-END
