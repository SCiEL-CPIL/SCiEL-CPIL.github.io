# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Strategy:
    1. Multi-start greedy with difference-pressure candidate scoring.
    2. Local search with row-rebuilds, multi-row rebuilds, and UCB-inspired exploration.
    3. Polish mode in final seconds for aggressive max reduction.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit
    polish_deadline = start_time + 103.0  # Start polish mode at ~103s (17s of polish)

    def time_up():
        return time.time() > deadline

    def polish_mode():
        return time.time() > polish_deadline

    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed

    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # Difference-pressure scoring: blend of compactness and collision risk
    def score_candidate(x, new_diffs, local_used, forbidden, current_row_max_target):
        """
        Score a candidate: lower is better.
        Blends the value x (compactness) with max new diff (collision pressure).
        """
        # Compactness: prefer smaller x
        compactness = x
        # Collision pressure: prefer candidates whose new diffs are smaller
        # (less likely to conflict with future rows)
        max_new_diff = max(new_diffs) if new_diffs else 0
        # Blend: 0.7 * compactness + 0.3 * max_new_diff
        # This encourages compact rows that don't introduce large differences
        score = 0.7 * compactness + 0.3 * max_new_diff
        return score

    # 1. Multi-start Greedy with difference-pressure scoring
    def greedy_construct(seed, target_max=None, use_pressure=True):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for pos in range(k):
                last = row[-1]
                remaining = k - pos - 1
                
                if target_max is not None:
                    max_x_allowed = target_max - remaining
                else:
                    max_x_allowed = 50000
                
                if last + 1 > max_x_allowed:
                    return None
                
                valid_candidates = []
                width = 12
                
                for x in range(last + 1, max_x_allowed + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        if use_pressure:
                            sc = score_candidate(x, new_diffs, local_used, used, target_max or max_x_allowed)
                        else:
                            sc = x
                        valid_candidates.append((x, new_diffs, sc))
                        if len(valid_candidates) >= width:
                            break
                
                if not valid_candidates:
                    return None
                
                # Selection: weighted by score (lower score = higher probability)
                # Use softmax-like selection with temperature
                if len(valid_candidates) == 1:
                    x, new_diffs, _ = valid_candidates[0]
                else:
                    # Prefer lower scores but allow exploration
                    scores = [c[2] for c in valid_candidates]
                    min_sc = min(scores)
                    # Exponential weights: lower score gets higher weight
                    weights = [math.exp(-(s - min_sc) / (min_sc * 0.1 + 1)) for s in scores]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    idx = 0
                    for i, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            idx = i
                            break
                    x, new_diffs, _ = valid_candidates[idx]
                
                row.append(x)
                local_used.update(new_diffs)
            
            rows.append(row)
            used.update(local_used)
        
        return rows, used

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used = None
    
    # Multi-start: try several seeds with and without pressure
    for seed in range(20):
        if time_up():
            break
        use_pressure = (seed % 2 == 0)  # Alternate pressure on/off for diversity
        res = greedy_construct(seed, use_pressure=use_pressure)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = [r[:] for r in rows]
                current_used = set(used)
    
    # Second pass: try with target max = best_max - 1
    if best_rows is not None and best_max > lb:
        for seed in range(15):
            if time_up():
                break
            target = best_max - 1
            if target <= k:
                break
            res = greedy_construct(seed, target, use_pressure=True)
            if res:
                rows, used = res
                m = get_max(rows)
                if m < best_max:
                    best_max = m
                    best_rows = [r[:] for r in rows]
                    current_rows = [r[:] for r in rows]
                    current_used = set(used)
    
    if best_rows is None:
        res = greedy_construct(0, use_pressure=False)
        if res:
            best_rows, current_used = res
            current_rows = [r[:] for r in best_rows]
            best_max = get_max(best_rows)
        else:
            return []

    if best_max == lb:
        return best_rows

    # 2. Local Search with Rebuilds
    rng = random.Random(12345)
    current_max = best_max

    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, bias=0.3, temperature=1.0, use_pressure=True):
        """
        Builds a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_diffs, with max entry <= max_val_limit.
        """
        row = [0]
        local_used = set()
        
        for pos in range(k):
            last = row[-1]
            valid_candidates = []
            
            remaining = k - pos - 1
            max_x_allowed = max_val_limit - remaining
            
            if last + 1 > max_x_allowed:
                return None
            
            width = 15 if temperature > 0.5 else 10
            
            for x in range(last + 1, max_x_allowed + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    if use_pressure:
                        sc = score_candidate(x, new_diffs, local_used, forbidden_diffs, max_val_limit)
                    else:
                        sc = x
                    valid_candidates.append((x, new_diffs, sc))
                    if len(valid_candidates) >= width:
                        break
            
            if not valid_candidates:
                return None
            
            # Selection with temperature and bias
            if len(valid_candidates) == 1:
                x, new_diffs, _ = valid_candidates[0]
            else:
                if temperature > 0.5 and rng.random() < bias:
                    # Weighted random: prefer lower scores but allow exploration
                    scores = [c[2] for c in valid_candidates]
                    min_sc = min(scores)
                    # Higher temperature = flatter distribution
                    temp_factor = max_sc_range = (max(scores) - min_sc) * 0.1 * temperature + 1
                    weights = [math.exp(-(s - min_sc) / temp_factor) for s in scores]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    idx = 0
                    for i, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            idx = i
                            break
                    x, new_diffs, _ = valid_candidates[idx]
                elif bias > 0 and rng.random() < bias:
                    idx = rng.randint(0, len(valid_candidates) - 1)
                    x, new_diffs, _ = valid_candidates[idx]
                else:
                    # Pick the best-scoring candidate
                    best_idx = min(range(len(valid_candidates)), key=lambda i: valid_candidates[i][2])
                    x, new_diffs, _ = valid_candidates[best_idx]
            
            row.append(x)
            local_used.update(new_diffs)
        
        return row

    def rebuild_two_rows(idx1, idx2, forbidden_diffs, rng, max_val_limit, temperature=1.0):
        """
        Attempt to rebuild two rows simultaneously.
        Returns (row1, row2) or (None, None).
        """
        # Try building row1 first, then row2 with row1's diffs added to forbidden
        row1 = rebuild_row(k, forbidden_diffs, rng, max_val_limit, bias=0.5, temperature=temperature)
        if row1 is None:
            return None, None
        row1_diffs = get_diffs(row1)
        forbidden2 = set(forbidden_diffs)
        forbidden2.update(row1_diffs)
        row2 = rebuild_row(k, forbidden2, rng, max_val_limit, bias=0.5, temperature=temperature)
        if row2 is None:
            return None, None
        return row1, row2

    no_improve_count = 0
    max_no_improve = 40
    # UCB-inspired: track exploration vs exploitation
    explore_count = 0
    exploit_count = 0

    def get_max_rows(rows, current_max):
        max_rows = []
        for i, row in enumerate(rows):
            if max(row) == current_max:
                max_rows.append(i)
        return max_rows

    base_temp = 1.0
    decay_rate = 0.997

    while not time_up():
        in_polish = polish_mode()
        
        # UCB-inspired exploration: when stuck, increase exploration
        if no_improve_count > 10:
            explore_prob = 0.5
        else:
            explore_prob = 0.2
        
        # In polish mode, focus on improvement
        if in_polish:
            max_rows = get_max_rows(current_rows, current_max)
            if max_rows:
                i = rng.choice(max_rows)
            else:
                i = rng.randint(0, n_rows - 1)
        else:
            max_rows = get_max_rows(current_rows, current_max)
            if max_rows and rng.random() < 0.6:
                i = rng.choice(max_rows)
            else:
                i = rng.randint(0, n_rows - 1)
        
        old_row = current_rows[i]
        old_diffs = get_diffs(old_row)
        
        current_used.difference_update(old_diffs)
        
        new_row = None
        current_temp = base_temp * (decay_rate ** no_improve_count)
        
        # In polish mode: only try to improve
        if in_polish:
            if max(old_row) == current_max:
                target_max = current_max - 1
                if target_max > k:
                    new_row = rebuild_row(k, current_used, rng, target_max, bias=0.2, temperature=0.3, use_pressure=True)
        else:
            # Attempt 1: Improve if this row holds the max
            if max(old_row) == current_max:
                target_max = current_max - 1
                if target_max > k:
                    new_row = rebuild_row(k, current_used, rng, target_max, bias=0.4, temperature=current_temp, use_pressure=True)
            
            # Attempt 2: Multi-row rebuild to escape local minima
            if new_row is None and not in_polish and rng.random() < 0.15:
                # Try rebuilding this row and a random other row together
                other_candidates = [j for j in range(n_rows) if j != i]
                if other_candidates:
                    j = rng.choice(other_candidates)
                    old_row2 = current_rows[j]
                    old_diffs2 = get_diffs(old_row2)
                    current_used.difference_update(old_diffs2)
                    
                    # Try to rebuild both with current_max (reshuffle)
                    r1, r2 = rebuild_two_rows(i, j, current_used, rng, current_max, temperature=min(1.0, current_temp * 1.3))
                    
                    if r1 is not None:
                        new_row = r1
                        # Also update the other row
                        current_rows[j] = r2
                        new_diffs2 = get_diffs(r2)
                        current_used.update(new_diffs2)
                    else:
                        # Restore
                        current_used.update(old_diffs2)
            
            # Attempt 3: Reshuffle (maintain max)
            if new_row is None:
                temp_for_reshuffle = min(1.0, current_temp * 1.5)
                new_row = rebuild_row(k, current_used, rng, current_max, bias=0.7, temperature=temp_for_reshuffle, use_pressure=True)
        
        if new_row is None:
            current_used.update(old_diffs)
            no_improve_count += 1
            explore_count += 1
        else:
            current_rows[i] = new_row
            new_diffs = get_diffs(new_row)
            current_used.update(new_diffs)
            
            new_global_max = get_max(current_rows)
            if new_global_max < current_max:
                current_max = new_global_max
                no_improve_count = 0
                base_temp = 1.0
                exploit_count += 1
            else:
                no_improve_count += 1
                explore_count += 1
        
        if no_improve_count > max_no_improve:
            # Aggressive restart
            no_improve_count = 0
            base_temp = 1.0
            
            if in_polish:
                # In polish mode: try to rebuild all max-holding rows with tighter target
                for _ in range(5):
                    if time_up():
                        break
                    max_rows = get_max_rows(current_rows, current_max)
                    if not max_rows:
                        break
                    i = rng.choice(max_rows)
                    old_row = current_rows[i]
                    old_diffs = get_diffs(old_row)
                    current_used.difference_update(old_diffs)
                    
                    target_max = current_max - 1
                    if target_max > k:
                        new_row = rebuild_row(k, current_used, rng, target_max, bias=0.1, temperature=0.2, use_pressure=True)
                        if new_row is not None:
                            current_rows[i] = new_row
                            current_used.update(get_diffs(new_row))
                            new_global_max = get_max(current_rows)
                            if new_global_max < current_max:
                                current_max = new_global_max
                        else:
                            current_used.update(old_diffs)
            else:
                # Normal mode: rebuild multiple rows
                for _ in range(4):
                    if time_up():
                        break
                    max_rows = get_max_rows(current_rows, current_max)
                    if not max_rows:
                        break
                    i = rng.choice(max_rows)
                    old_row = current_rows[i]
                    old_diffs = get_diffs(old_row)
                    current_used.difference_update(old_diffs)
                    
                    target_max = current_max - 1
                    if target_max > k:
                        new_row = rebuild_row(k, current_used, rng, target_max, bias=0.3, temperature=1.0, use_pressure=True)
                        if new_row is not None:
                            current_rows[i] = new_row
                            current_used.update(get_diffs(new_row))
                            new_global_max = get_max(current_rows)
                            if new_global_max < current_max:
                                current_max = new_global_max
                        else:
                            current_used.update(old_diffs)
    
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True

    # Track best
    if validate(current_rows, n_rows, k):
        if current_max < best_max:
            best_max = current_max
            best_rows = [r[:] for r in current_rows]

    # Final polish: try to improve with tighter target
    if best_rows is not None and best_max > lb:
        for _ in range(10):
            if time_up():
                break
            target = best_max - 1
            if target <= k:
                break
            new_rows = []
            new_used = set()
            success = True
            for r in best_rows:
                row_diffs = get_diffs(r)
                new_used.difference_update(row_diffs)
                new_row = rebuild_row(k, new_used, rng, target, bias=0.1, temperature=0.3, use_pressure=True)
                if new_row is None:
                    success = False
                    break
                new_rows.append(new_row)
                new_used.update(get_diffs(new_row))
            
            if success and validate(new_rows, n_rows, k):
                new_max = get_max(new_rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = new_rows

    if not validate(best_rows, n_rows, k):
        return [[0] * (k + 1)] * n_rows
    
    return best_rows
# EVOLVE-BLOCK-END
