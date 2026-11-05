import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    lower_bound = int(math.ceil(math.sqrt(2 * total_diffs))) - 1

    best_scope = float('inf')
    best_rows = None

    # Phase 1: Multiple randomized greedy constructions with different strategies
    seed = 0
    while time.time() < deadline - 40:
        # Alternate between different greedy strategies
        strategy = seed % 3
        
        if strategy == 0:
            rows = _greedy_packing(n_rows, k, seed)
        elif strategy == 1:
            rows = _greedy_spread(n_rows, k, seed)
        else:
            rows = _greedy_balanced(n_rows, k, seed)
        
        # Calculate scope
        scope = _scope(rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
            if best_scope <= lower_bound:
                break
        
        seed += 1
    
    if best_rows is None:
        best_rows = _greedy_packing(n_rows, k, 0)
        best_scope = _scope(best_rows)

    # Phase 2: Local search / simulated annealing to improve
    if time.time() < deadline - 10:
        best_rows, best_scope = _local_search(best_rows, n_rows, k, time.time(), deadline - 5)
    
    return best_rows

def _scope(rows):
    return max(max(r) for r in rows)

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

def _greedy_packing(n_rows, k, seed):
    """Greedy that packs rows tightly, preferring smaller values."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            # Find the smallest valid x
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

def _greedy_spread(n_rows, k, seed):
    """Greedy that spreads out differences, preferring larger gaps."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for pos in range(k):
            # Try several candidates and pick one that maximizes minimum gap to used diffs
            best_x = None
            best_score = -1
            last_x = row[-1]
            
            # Look at a range of candidates
            for x in range(last_x + 1, last_x + 30):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                
                # Score: minimum distance from new diffs to used diffs
                if used:
                    min_dist = min(min(abs(d - u) for u in used) for d in new_diffs)
                else:
                    min_dist = x
                
                if min_dist > best_score:
                    best_score = min_dist
                    best_x = x
            
            if best_x is None:
                # Fallback: find any valid x
                x = last_x + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        best_x = x
                        break
                    x += 1
            
            row.append(best_x)
            new_diffs = {best_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _greedy_balanced(n_rows, k, seed):
    """Greedy with a balanced heuristic: mix of packing and spreading."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for pos in range(k):
            last_x = row[-1]
            
            # Collect valid candidates in a window
            valid_xs = []
            for x in range(last_x + 1, last_x + 50):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
            
            if not valid_xs:
                x = last_x + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        break
                    x += 1
            
            # Score each candidate
            scores = []
            for x in valid_xs:
                new_diffs = {x - a for a in row}
                if used:
                    # Penalize closeness to used diffs
                    closeness = sum(min(abs(d - u) for u in used) for d in new_diffs) / len(new_diffs)
                    # Reward larger values (spreading)
                    spread_bonus = x * 0.1
                    score = closeness + spread_bonus
                else:
                    score = x
                
                scores.append(score)
            
            # Pick the candidate with the best score (highest = better spread)
            # Add some randomness
            if len(valid_xs) > 1 and rng.random() < 0.5:
                # Weighted random selection
                max_score = max(scores)
                min_score = min(scores)
                if max_score > min_score:
                    weights = [s - min_score + 1 for s in scores]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cumulative = 0
                    chosen_idx = 0
                    for i, w in enumerate(weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen_idx = i
                            break
                else:
                    chosen_idx = rng.randint(0, len(valid_xs) - 1)
            else:
                # Pick the best scoring candidate
                chosen_idx = scores.index(max(scores))
            
            chosen_x = valid_xs[chosen_idx]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _local_search(rows, n_rows, k, start_time, deadline):
    """Local search to reduce scope by moving elements to smaller positions."""
    rng = random.Random(int(start_time * 1000) % 100000)
    
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    # Temperature for simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Pick a random row and position
        row_idx = rng.randint(0, n_rows - 1)
        pos_idx = rng.randint(1, k)
        
        old_val = current_rows[row_idx][pos_idx]
        
        # Determine valid range for new value
        lower = current_rows[row_idx][pos_idx - 1] + 1
        upper = current_rows[row_idx][pos_idx + 1] - 1 if pos_idx < k else old_val
        
        if upper < lower:
            T *= cooling
            continue
        
        # Try a few candidate values, biased towards smaller values
        num_tries = min(5, upper - lower + 1)
        for _ in range(num_tries):
            if new_val >= old_val:
                continue
            
            # Temporarily update
            old = current_rows[row_idx][pos_idx]
            current_rows[row_idx][pos_idx] = new_val
            
            if _is_valid(current_rows):
                new_scope = _scope(current_rows)
                delta = new_scope - current_scope
                
                # Accept if better or with probability exp(-delta/T)
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                    break
                else:
                    current_rows[row_idx][pos_idx] = old
            else:
                current_rows[row_idx][pos_idx] = old
        
        T *= cooling
    
    return best_rows, best_scope
