import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 110
    
    # Initial solution via randomized greedy with multiple restarts
    best_solution = None
    best_score = float('inf')
    
    # Try a few different greedy initializations
    for attempt in range(20):
        random.seed(attempt * 137 + 42)
        sol = _greedy_construct_randomized(n_rows, k)
        if sol:
            score = max(max(r) for r in sol)
            if score < best_score:
                best_score = score
                best_solution = sol
        if time.time() > deadline - 30:
            break
    
    if not best_solution:
        # Fallback: very aggressive greedy
        best_solution = _greedy_construct(n_rows, k)
        if not best_solution:
            return []
        best_score = max(max(r) for r in best_solution)
    
    # Simulated annealing with multiple move types
    best_solution = _simulated_annealing(best_solution, n_rows, k, deadline)
    
    # Final local search polish
    if time.time() < deadline - 2:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline)
        if polished:
            pol_score = max(max(r) for r in polished)
            if pol_score < best_score:
                best_solution = polished
                best_score = pol_score
    
    return best_solution

def _greedy_construct(n_rows, k):
    """Basic greedy: pick smallest valid value for each position."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _greedy_construct_randomized(n_rows, k):
    """Greedy with randomization in value selection."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            # Try a window of candidates, pick randomly among valid ones
            window_start = current_max + 1
            window_end = current_max + 50 + random.randint(0, 100)
            valid_candidates = []
            for x in range(window_start, window_end + 1):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    valid_candidates.append((x, new_diffs))
            
            if not valid_candidates:
                # Expand window
                for x in range(window_end + 1, window_end + 500):
                    new_diffs = []
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        valid_candidates.append((x, new_diffs))
                        break
            
            if not valid_candidates:
                return None
            
            # Pick the smallest or a random one
            if random.random() < 0.7:
                x, new_diffs = valid_candidates[0]
            else:
                x, new_diffs = random.choice(valid_candidates)
            
            row.append(x)
            used_diffs.update(new_diffs)
            current_max = x
        rows.append(row)
    return rows

def _build_diff_set(rows, n):
    """Build the set of all differences."""
    diffs = set()
    for r in rows:
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(r[j] - r[i])
    return diffs

def _check_validity(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    n = k + 1
    all_diffs = set()
    for r in rows:
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def _simulated_annealing(initial_rows, n_rows, k, deadline):
    """Simulated annealing to minimize the maximum entry."""
    n = k + 1
    rows = [r[:] for r in initial_rows]
    best_rows = [r[:] for r in rows]
    best_score = max(max(r) for r in rows)
    
    # Temperature schedule
    t_start = 50.0
    t_end = 0.1
    cooling_rate = 0.995
    
    current_temp = t_start
    
    while time.time() < deadline - 5 and current_temp > t_end:
        # Pick a random move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Single position change
            new_rows, delta = _sa_move_single_pos(rows, n_rows, k, n)
        elif move_type < 0.7:
            # Row re-solve: regenerate one row given the others' diffs
            new_rows, delta = _sa_move_resolve_row(rows, n_rows, k, n)
        elif move_type < 0.85:
            # Swap two rows
            new_rows, delta = _sa_move_row_swap(rows, n_rows, k, n)
        else:
            # Multi-position change in one row
            new_rows, delta = _sa_move_multi_pos(rows, n_rows, k, n)
        
        if new_rows is None:
            current_temp *= cooling_rate
            continue
        
        new_score = max(max(r) for r in new_rows)
        old_score = max(max(r) for r in rows)
        
        # Accept if better or with probability exp(-delta/temp)
        if new_score < old_score:
            rows = new_rows
            if new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in rows]
        else:
            delta_score = new_score - old_score
            if delta_score == 0:
                rows = new_rows
            else:
                prob = _safe_exp(-delta_score / current_temp)
                if random.random() < prob:
                    rows = new_rows
        
        current_temp *= cooling_rate
        
        # Occasionally reset to best to avoid getting stuck
        if random.random() < 0.01:
            rows = [r[:] for r in best_rows]
    
    return best_rows

def _safe_exp(x):
    """Safe exponential that handles very negative values."""
    if x < -100:
        return 0.0
    if x > 100:
        return 1.0
    return 2.718281828459045 ** x

def _sa_move_single_pos(rows, n_rows, k, n):
    """Try to change one position in a random row."""
    r_idx = random.randint(0, n_rows - 1)
    pos = random.randint(1, k)
    
    row = rows[r_idx]
    old_val = row[pos]
    
    # Build diff set
    all_diffs = _build_diff_set(rows, n)
    
    # Diffs involving old_val
    diffs_involving = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving.add(old_val - row[i])
        else:
            diffs_involving.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving
    
    # Try a range of new values
    lower_bound = row[pos - 1] + 1
    upper_bound = old_val + random.randint(5, 30)
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos + 1] - 1)
    
    # Also try smaller values
    lower_bound = max(row[pos - 1] + 1, old_val - random.randint(5, 30))
    
    best_new_val = None
    best_new_score = float('inf')
    
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if valid:
            # Compute new score
            new_row = row[:]
            new_row[pos] = new_val
            new_max = max(new_row)
            # Check if this reduces the overall max
            other_max = max(max(r) for i, r in enumerate(rows) if i != r_idx)
            new_score = max(new_max, other_max)
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_new_val = new_val
    
    if best_new_val is None:
        return None, 0
    
    new_rows = [r[:] for r in rows]
    new_rows[r_idx][pos] = best_new_val
    old_score = max(max(r) for r in rows)
    new_score = max(max(r) for r in new_rows)
    return new_rows, new_score - old_score

def _sa_move_resolve_row(rows, n_rows, k, n):
    """Regenerate one row given the differences used by other rows."""
    r_idx = random.randint(0, n_rows - 1)
    
    # Collect diffs from all other rows
    other_diffs = set()
    for i, r in enumerate(rows):
        if i == r_idx:
            continue
        for a in range(n):
            for b in range(a + 1, n):
                other_diffs.add(r[b] - r[a])
    
    # Try to build a new row avoiding other_diffs
    row = [0]
    current_max = 0
    success = True
    while len(row) < n:
        found = False
        for x in range(current_max + 1, current_max + 200):
            valid = True
            for a in row:
                d = x - a
                if d in other_diffs:
                    valid = False
                    break
            if valid:
                row.append(x)
                current_max = x
                found = True
                break
        if not found:
            success = False
            break
    
    if not success:
        return None, 0
    
    # Check if the new row's diffs don't conflict with itself
    row_diffs = set()
    for a in range(n):
        for b in range(a + 1, n):
            d = row[b] - row[a]
            if d in row_diffs:
                return None, 0
            row_diffs.add(d)
    
    new_rows = [r[:] for r in rows]
    new_rows[r_idx] = row
    old_score = max(max(r) for r in rows)
    new_score = max(max(r) for r in new_rows)
    return new_rows, new_score - old_score

def _sa_move_row_swap(rows, n_rows, k, n):
    """Swap two rows."""
    if n_rows < 2:
        return None, 0
    
    r1, r2 = random.sample(range(n_rows), 2)
    new_rows = [r[:] for r in rows]
    new_rows[r1], new_rows[r2] = new_rows[r2], new_rows[r1]
    
    # Check validity (swapping rows doesn't change the set of diffs, so it's always valid)
    old_score = max(max(r) for r in rows)
    new_score = max(max(r) for r in new_rows)
    return new_rows, new_score - old_score

def _sa_move_multi_pos(rows, n_rows, k, n):
    """Change two positions in the same row."""
    r_idx = random.randint(0, n_rows - 1)
    pos1, pos2 = random.sample(range(1, k + 1), 2)
    if pos2 < pos1:
        pos1, pos2 = pos2, pos1
    
    row = rows[r_idx]
    v1, v2 = row[pos1], row[pos2]
    
    # Try small perturbations
    for dv1 in range(-3, 4):
        for dv2 in range(-3, 4):
            nv1, nv2 = v1 + dv1, v2 + dv2
            if nv1 <= row[pos1 - 1] or nv2 >= row[pos2 + 1] if pos2 + 1 < n else False:
                continue
            if nv2 <= row[pos1] and pos2 > pos1:  # simplified check
                continue
            if nv1 >= row[pos2] and pos1 < pos2:
                continue
            if nv1 <= 0 or nv2 <= 0:
                continue
            if pos1 + 1 < n and nv1 >= row[pos1 + 1]:
                continue
            if pos2 + 1 < n and nv2 >= row[pos2 + 1]:
                continue
            
            # Full validity check
            new_row = row[:]
            new_row[pos1] = nv1
            new_row[pos2] = nv2
            
            # Check monotonicity
            for i in range(n - 1):
                if new_row[i] >= new_row[i + 1]:
                    break
            else:
                # Check diffs
                all_diffs = _build_diff_set(rows, n)
                # Remove old diffs for this row
                for a in range(n):
                    for b in range(a + 1, n):
                        all_diffs.discard(row[b] - row[a])
                
                valid = True
                new_row_diffs = set()
                for a in range(n):
                    for b in range(a + 1, n):
                        d = new_row[b] - new_row[a]
                        if d in all_diffs or d in new_row_diffs:
                            valid = False
                            break
                        new_row_diffs.add(d)
                
                if valid:
                    new_rows = [r[:] for r in rows]
                    new_rows[r_idx] = new_row
                    old_score = max(max(r) for r in rows)
                    new_score = max(max(r) for r in new_rows)
                    return new_rows, new_score - old_score
    
    return None, 0

def _local_search_minimize(rows, n_rows, k, deadline):
    """Greedy local search: repeatedly try to reduce each position."""
    n = k + 1
    best_rows = [r[:] for r in rows]
    
    while time.time() < deadline - 3:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k + 1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break
            
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Build diff set
            all_diffs = _build_diff_set(best_rows, n)
            
            # Diffs involving old_val
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    diffs_involving.add(old_val - row[i])
                else:
                    diffs_involving.add(row[i] - old_val)
            
            forbidden_diffs = all_diffs - diffs_involving
            
            # Try to find a smaller valid value
            lower_bound = row[pos - 1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos + 1] - 1)
            
            if upper_bound >= lower_bound:
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        break
        
        if not progress:
            break
    
    return best_rows

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """Find a random valid value for a position."""
    n = k + 1
    row = rows[r_idx]
    old_val = row[pos]
    
    all_diffs = _build_diff_set(rows, n)
    
    diffs_involving = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving.add(old_val - row[i])
        else:
            diffs_involving.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving
    
    lower_bound = row[pos - 1] + 1
    upper_bound = 300
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos + 1] - 1)
    
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            valid_candidates.append(new_val)
    
    if not valid_candidates:
        return None
    return random.choice(valid_candidates)

def _local_search_ilS(rows, n_rows, k, deadline):
    """Deprecated alias."""
    return _local_search_minimize(rows, n_rows, k, deadline)

def _repair(rows, n_rows, k):
    """Attempt to repair invalid rows."""
    return None
