import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, randomize=False, seed=None):
    """
    Greedy initialization with optional randomization for diversity.
    """
    if seed is not None:
        random.seed(seed)
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if randomize and random.random() < 0.1:
                # Try a few extra values for diversity
                for xx in range(x, x + 3):
                    new_diffs = {xx - a for a in row}
                    if not new_diffs & used:
                        x = xx
                        break
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _compute_new_diffs(cand_row, pi, old_diffs):
    """
    Compute new diffs for a row after changing position pi.
    Only checks diffs involving position pi against the unchanged diffs.
    Returns (new_diffs_set, is_valid_internal).
    """
    n = len(cand_row)
    new_val = cand_row[pi]
    
    # Start with old diffs, remove those involving pi, add new ones
    new_diffs = set()
    
    # Add diffs NOT involving pi (unchanged)
    for i in range(n):
        if i == pi:
            continue
        for j in range(i + 1, n):
            if j == pi:
                continue
            new_diffs.add(cand_row[j] - cand_row[i])
    
    # Add diffs involving pi
    for i in range(n):
        if i == pi:
            continue
        d = abs(cand_row[pi] - cand_row[i])
        if d in new_diffs:
            return None, False
        new_diffs.add(d)
    
    return new_diffs, True

def _run_sa_chain(rows, n_rows, k, time_budget, T_start=200.0, T_min=0.3, cool=0.99997):
    """
    Run a single SA chain for the given time budget.
    Returns (best_rows, best_val).
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    row_maxes = [max(r) for r in rows]
    
    current_rows = [r[:] for r in rows]
    current_val = best_val
    T = T_start
    
    no_improve = 0
    max_no_improve = 20000
    
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        candidates = []
        
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(60, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
        max_increase = min(20, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        if random.random() < 0.2:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            old_diffs = row_diffs[ri]
            new_diffs, valid_internal = _compute_new_diffs(cand_row, pi, old_diffs)
            
            if not valid_internal:
                continue
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.3:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        if no_improve > 10000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = T_start
            no_improve = 0
    
    return best_rows, best_val

def _polish(rows, n_rows, k, time_budget):
    """
    Targeted polishing: focus on reducing the max entry.
    Returns (best_rows, best_val).
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    row_maxes = [max(r) for r in rows]
    row_diffs = [_row_diffs_set(r) for r in rows]
    
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    no_improve = 0
    max_no_improve = 15000
    
    while time.time() < deadline and no_improve < max_no_improve:
        if max_row_indices and random.random() < 0.7:
            ri = random.choice(max_row_indices)
        else:
            ri = random.randrange(n_rows)
        
        if ri in max_row_indices and random.random() < 0.5:
            pi = k
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        
        if max_decrease <= 0:
            no_improve += 1
            continue
        
        candidates = [old_val - d for d in range(max_decrease, 0, -1)]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            old_diffs = row_diffs[ri]
            new_diffs, valid_internal = _compute_new_diffs(cand_row, pi, old_diffs)
            
            if not valid_internal:
                continue
            
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.1
            else:
                accept = False
            
            if accept:
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
    
    return best_rows, best_val

def _final_local_search(rows, n_rows, k, time_budget):
    """
    Final high-precision local search: try all possible single-position
    decreases on all rows, accepting only strict improvements.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    row_maxes = [max(r) for r in rows]
    row_diffs = [_row_diffs_set(r) for r in rows]
    
    current_rows = [r[:] for r in rows]
    current_val = best_val
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for ri in range(n_rows):
            if time.time() >= deadline:
                break
            
            for pi in range(1, k + 1):
                if time.time() >= deadline:
                    break
                    
                old_val = current_rows[ri][pi]
                min_allowed = current_rows[ri][pi - 1] + 1
                max_decrease = old_val - min_allowed
                
                if max_decrease <= 0:
                    continue
                
                # Try all possible decreases from largest to smallest
                for d in range(max_decrease, 0, -1):
                    new_val = old_val - d
                    
                    if new_val <= current_rows[ri][pi - 1]:
                        continue
                    if pi < k and new_val >= current_rows[ri][pi + 1]:
                        continue
                        
                    cand_row = current_rows[ri][:]
                    cand_row[pi] = new_val
                    
                    old_diffs = row_diffs[ri]
                    new_diffs, valid_internal = _compute_new_diffs(cand_row, pi, old_diffs)
                    
                    if not valid_internal:
                        continue
                    
                    conflict = False
                    for diff in new_diffs:
                        if diff in all_used and diff not in old_diffs:
                            conflict = True
                            break
                    
                    if conflict:
                        continue
                    
                    cand_row_max = max(cand_row)
                    old_row_max = row_maxes[ri]
                    row_maxes[ri] = cand_row_max
                    new_global_max = max(row_maxes)
                    row_maxes[ri] = old_row_max
                    
                    if new_global_max < current_val:
                        all_used -= old_diffs
                        all_used |= new_diffs
                        current_rows[ri] = cand_row
                        row_diffs[ri] = new_diffs
                        row_maxes[ri] = cand_row_max
                        current_val = new_global_max
                        
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                        
                        improved = True
                        break  # Move on to next position after improvement
    
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    total_deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Strategy: Run 2 SA chains with different initializations, then polish
    # Time allocation: ~50s for each SA chain, ~15s polish, ~30s final local search
    
    # Chain 1: Standard greedy init
    rows1 = _greedy_init(n_rows, k, randomize=False)
    time_for_sa1 = 45.0
    best1, val1 = _run_sa_chain(rows1, n_rows, k, time_for_sa1)
    
    elapsed = time.time() - start_time
    remaining = total_deadline - start_time - elapsed
    
    # Chain 2: Randomized greedy init (if time permits)
    if remaining > 60.0:
        rows2 = _greedy_init(n_rows, k, randomize=True, seed=random.randint(0, 10000))
        time_for_sa2 = min(40.0, remaining - 50.0)
        if time_for_sa2 > 10.0:
            best2, val2 = _run_sa_chain(rows2, n_rows, k, time_for_sa2)
        else:
            best2, val2 = best1, val1
    else:
        best2, val2 = best1, val1
    
    # Pick the best from both chains
    if val2 < val1:
        best_rows = best2
        best_val = val2
    else:
        best_rows = best1
        best_val = val1
    
    elapsed = time.time() - start_time
    remaining = total_deadline - start_time - elapsed
    
    # Polish phase
    if remaining > 20.0:
        polish_time = min(20.0, remaining - 10.0)
        if polish_time > 5.0:
            best_rows, best_val = _polish(best_rows, n_rows, k, polish_time)
    
    elapsed = time.time() - start_time
    remaining = total_deadline - start_time - elapsed
    
    # Final local search
    if remaining > 5.0:
        local_time = remaining - 2.0
        if local_time > 2.0:
            best_rows, best_val = _final_local_search(best_rows, n_rows, k, local_time)
    
    return best_rows

# EVOLVE-BLOCK-END
