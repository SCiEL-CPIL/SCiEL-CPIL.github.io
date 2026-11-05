import time
import random
import bisect
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    best_scope = float('inf')
    best_rows = None

    # Candidate pool size for the greedy selection
    c = 12

    seed = 0

    while time.time() < deadline - 30:
        rows = _greedy_construct(n_rows, k, seed, c)
        
        # Try to polish with simulated annealing
        if time.time() < deadline - 15:
            rows = _simulated_annealing(rows, n_rows, k, time.time(), deadline - 5)
        
        # Calculate scope
        scope = 0
        for row in rows:
            if row:
                scope = max(scope, row[-1])
        
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            
            # If we hit the lower bound, we can't do better
            if best_scope == lower_bound:
                break
        
        seed += 1

    # Final polish pass if time permits
    if best_rows is not None and time.time() < deadline - 5:
        best_rows = _polish_max(best_rows, n_rows, k, time.time(), deadline - 1)
        best_scope = max(max(r) for r in best_rows)

    # If no solution found (should not happen), return simple greedy
    if best_rows is None:
        best_rows = _simple_greedy(n_rows, k)
        
    return best_rows

def _scope(rows):
    return max(max(r) for r in rows) if rows else 0

def _all_diffs(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used

def _is_valid(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _packing_score(new_diffs, used_sorted):
    """Score a candidate: sum of distances to nearest used diff on each side.
    Higher score means more 'space' around the new diffs."""
    if not used_sorted:
        return 0
    score = 0
    for d in new_diffs:
        # Find insertion point in sorted used
        idx = bisect.bisect_left(used_sorted, d)
        # Distance to left neighbor
        if idx > 0:
            score += d - used_sorted[idx - 1]
        else:
            score += d
        # Distance to right neighbor
        if idx < len(used_sorted):
            score += used_sorted[idx] - d
        else:
            score += d
    return score

def _greedy_construct(n_rows, k, seed, c):
    """Greedy construction with packing-based heuristic."""
    rng = random.Random(seed)
    used = set()
    used_sorted = []
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            # Find valid candidates
            valid_xs = []
            scores = []
            last_x = row[-1]
            
            x = last_x + 1
            # Search up to a reasonable limit
            max_search = last_x + 200
            while x <= max_search and len(valid_xs) < c:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                    scores.append(_packing_score(new_diffs, used_sorted))
                x += 1
            
            # If not enough candidates, expand search
            if len(valid_xs) < c:
                x = last_x + 1
                while len(valid_xs) < c:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        scores.append(_packing_score(new_diffs, used_sorted))
                    x += 1
            
            if not valid_xs:
                # Fallback: find any valid x
                x = last_x + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        scores.append(0)
                        break
                    x += 1
            
            # Select candidate: prefer higher packing score with randomness
            if len(valid_xs) > 1:
                # Temperature-based selection
                max_s = max(scores)
                min_s = min(scores)
                if max_s > min_s:
                    # Softmax-like selection
                    temp = (max_s - min_s) / 3.0 + 1.0
                    exp_scores = [math.exp((s - max_s) / temp) for s in scores]
                    total = sum(exp_scores)
                    r = rng.random() * total
                    cumulative = 0
                    chosen_idx = 0
                    for i, e in enumerate(exp_scores):
                        cumulative += e
                        if r <= cumulative:
                            chosen_idx = i
                            break
                else:
                    chosen_idx = rng.randint(0, len(valid_xs) - 1)
            else:
                chosen_idx = 0
            
            chosen_x = valid_xs[chosen_idx]
            
            # Update row and used
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            # Keep used_sorted updated
            for d in new_diffs:
                bisect.insort(used_sorted, d)
            
        rows.append(row)
        
    return rows

def _simple_greedy(n_rows, k):
    """Simple greedy fallback."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used.update(new_diffs)
        rows.append(row)
    
    return rows

def _simulated_annealing(rows, n_rows, k, start_time, deadline):
    """Simulated annealing to reduce scope."""
    rng = random.Random(int(start_time * 10000) % 1000000)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    # Annealing parameters
    t0 = 5.0
    t_end = 0.1
    cooling = 0.97
    
    t = t0
    no_improve = 0
    
    while time.time() < deadline and t > t_end:
        # Pick a random element to modify
        row_idx = rng.randint(0, n_rows - 1)
        pos_idx = rng.randint(1, k)
        
        old_val = current_rows[row_idx][pos_idx]
        
        # Determine valid range for new value
        lower = current_rows[row_idx][pos_idx - 1] + 1
        upper = current_rows[row_idx][pos_idx + 1] - 1 if pos_idx < k else old_val - 1
        
        if upper <= lower:
            t *= cooling
            continue
        
        # Try to move to a smaller value (bias toward reduction)
        if rng.random() < 0.7:
            # Try values below current
            search_range = min(20, old_val - lower)
            if search_range > 0:
                new_val = rng.randint(lower, min(upper, old_val - 1))
            else:
                t *= cooling
                continue
        else:
            # Try any value in range
            new_val = rng.randint(lower, upper)
        
        if new_val == old_val:
            t *= cooling
            continue
        
        # Compute delta in scope
        old_max = current_rows[row_idx][pos_idx]
        current_rows[row_idx][pos_idx] = new_val
        
        if not _is_valid(current_rows):
            current_rows[row_idx][pos_idx] = old_val
            t *= cooling
            continue
        
        new_scope = _scope(current_rows)
        delta = new_scope - current_scope
        
        if delta < 0 or rng.random() < math.exp(-delta / t):
            current_scope = new_scope
            if delta < 0:
                no_improve = 0
            else:
                no_improve += 1
        else:
            current_rows[row_idx][pos_idx] = old_val
            no_improve += 1
        
        t *= cooling
        
        if no_improve > 100:
            break
    
    return current_rows

def _polish_max(rows, n_rows, k, start_time, deadline):
    """Targeted polish: try to reduce the maximum element."""
    rng = random.Random(int(start_time * 10000) % 1000000)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    # Find the row with the maximum element
    max_row_idx = 0
    for i, row in enumerate(current_rows):
        if row[-1] > current_rows[max_row_idx][-1]:
            max_row_idx = i
    
    max_pos = k  # The last element is the maximum
    
    old_val = current_rows[max_row_idx][max_pos]
    
    # Try to reduce it
    lower = current_rows[max_row_idx][max_pos - 1] + 1
    upper = old_val - 1
    
    if upper < lower:
        return current_rows
    
    # Try values from upper down to lower
    for new_val in range(upper, lower - 1, -1):
        if time.time() > deadline:
            break
        
        current_rows[max_row_idx][max_pos] = new_val
        
        if _is_valid(current_rows):
            new_scope = _scope(current_rows)
            if new_scope < current_scope:
                current_scope = new_scope
                break
        else:
            current_rows[max_row_idx][max_pos] = old_val
            continue
    
    # Also try reducing other elements in the max row
    for pos_idx in range(k - 1, 0, -1):
        if time.time() > deadline:
            break
        
        old_v = current_rows[max_row_idx][pos_idx]
        lower_v = current_rows[max_row_idx][pos_idx - 1] + 1
        upper_v = current_rows[max_row_idx][pos_idx + 1] - 1 if pos_idx < k else old_v - 1
        
        if upper_v < lower_v:
            continue
        
        for new_v in range(upper_v, lower_v - 1, -1):
            if new_v >= old_v:
                continue
            
            current_rows[max_row_idx][pos_idx] = new_v
            
            if _is_valid(current_rows):
                new_scope = _scope(current_rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    break
            else:
                current_rows[max_row_idx][pos_idx] = old_v
    
    return current_rows
