import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, rng=None, randomize=False):
    """
    Standard greedy initialization. Optionally randomize the choice of the next value.
    """
    if rng is None:
        rng = random
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                if randomize and rng.random() < 0.3 and x < 500:
                    # Try to jump ahead a bit to create variation
                    jump = rng.randint(1, 10)
                    x2 = x + jump
                    new_diffs2 = {x2 - a for a in row}
                    if not new_diffs2 & used:
                        row.append(x2)
                        used |= new_diffs2
                        x = x2
                        continue
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

def _build_static_diffs(row, k):
    """
    For each position pi (1..k), compute the set of differences in the row
    that do NOT involve position pi.
    """
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                s.add(row[j] - row[i])
        static[pi] = s
    return static

def _is_move_valid(row, pi, new_val, static_pi, old_diffs, all_used):
    """
    Fast validity check without creating the new_diffs set.
    Returns True if the move is valid.
    """
    n = len(row)
    new_diffs_list = []
    
    # Compute new diffs involving pi
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        new_diffs_list.append(d)
        
    # Check internal validity
    seen_new = set()
    for d in new_diffs_list:
        # Must not be in static diffs (which are unchanged)
        if d in static_pi:
            return False
        # Must not be duplicate among new diffs
        if d in seen_new:
            return False
        seen_new.add(d)
        
    # Check global conflicts
    # A conflict exists if a new diff is in all_used but was NOT in old_diffs
    for d in new_diffs_list:
        if d in all_used and d not in old_diffs:
            return False
            
    return True

def _get_new_diffs_set(row, pi, new_val, static_pi):
    """
    Construct the new_diffs set. Only called if move is accepted.
    """
    n = len(row)
    new_diffs = set(static_pi)
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        new_diffs.add(d)
    return new_diffs

def _random_valid_row(forbidden_diffs, max_limit, k, max_candidates=15, rng=None):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_diffs,
    with max entry <= max_limit.
    """
    if rng is None:
        rng = random
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for step in range(k):
        candidates = []
        x = last_val + 1
        # Wider search window for earlier steps
        search_width = 400 if step < k // 2 else 250
        limit = min(max_limit, last_val + search_width) if max_limit is not None else last_val + search_width
        
        while x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                candidates.append(x)
            
            x += 1
            if len(candidates) >= max_candidates:
                break
        
        if not candidates:
            return None
        
        # Weight toward smaller candidates to keep the row compact
        if len(candidates) > 1 and rng.random() < 0.6:
            # Pick from the first half of candidates
            half = max(1, len(candidates) // 2)
            chosen_x = rng.choice(candidates[:half])
        else:
            chosen_x = rng.choice(candidates)
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def _row_reconstruction_sa(rows, n_rows, k, deadline, rng):
    """
    Row-level simulated annealing: repeatedly replace one row with a random valid
    row, accepting worse solutions with probability based on temperature.
    """
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    T = 80.0
    T_min = 0.5
    cool = 0.999
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline - 15 and no_improve < max_no_improve:
        # Preferentially target the row that achieves the global max
        if rng.random() < 0.7:
            max_rows = [i for i in range(n_rows) if row_maxes[i] == current_val]
            if max_rows:
                ri = rng.choice(max_rows)
            else:
                max_rows = [i for i in range(n_rows) if row_maxes[i] >= current_val - 3]
                ri = rng.choice(max_rows) if max_rows else rng.randrange(n_rows)
        elif rng.random() < 0.5:
            # Target second-highest rows
            sorted_rows = sorted(range(n_rows), key=lambda i: row_maxes[i], reverse=True)
            ri = rng.choice(sorted_rows[:max(1, n_rows // 3)])
        else:
            ri = rng.randrange(n_rows)
        
        forbidden = set()
        for j in range(n_rows):
            if j != ri:
                forbidden |= row_diffs[j]
        
        # Target: try to beat current global max, or at least not worsen it
        target_max = current_val - 1
        if target_max < k:
            target_max = current_rows[ri][-1] - 1
        
        new_row = _random_valid_row(forbidden, target_max, k, max_candidates=15, rng=rng)
        
        if new_row is None:
            target_max2 = current_val
            new_row = _random_valid_row(forbidden, target_max2, k, max_candidates=12, rng=rng)
        
        if new_row is None:
            target_max3 = current_val + 20
            new_row = _random_valid_row(forbidden, target_max3, k, max_candidates=10, rng=rng)
        
        if new_row is None:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        new_row_max = new_row[-1]
        other_maxes = [row_maxes[j] for j in range(n_rows) if j != ri]
        new_global_max = max(max(other_maxes) if other_maxes else 0, new_row_max)
        
        delta = new_global_max - current_val
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if rng.random() < 0.5:
                accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta / max(T, 0.1))
                if rng.random() < prob:
                    accept = True
        
        if accept:
            new_diffs = _row_diffs_set(new_row)
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            row_maxes[ri] = new_row_max
            current_val = new_global_max
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        # Restart if stuck for too long
        if no_improve > max_no_improve // 3:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            row_maxes = [max(r) for r in current_rows]
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            T = 80.0
            no_improve = 0

    return [r[:] for r in best_rows]

def _element_sa(rows, n_rows, k, deadline, rng):
    """
    Element-level simulated annealing with targeted moves.
    """
    current_rows = [r[:] for r in rows]
    all_used = _compute_all_used(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_val = _global_max(current_rows)
    best_sum_max = sum(max(r) for r in best_rows)
    
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    
    T = 200.0
    T_min = 0.5
    cool = 0.99995
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline - 15.0:
        # Targeted row selection
        if rng.random() < 0.6:
            max_rows = [i for i in range(n_rows) if row_maxes[i] == best_val]
            if max_rows:
                ri = rng.choice(max_rows)
            else:
                ri = rng.randrange(n_rows)
        else:
            ri = rng.randrange(n_rows)
        
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        if rng.random() < 0.15:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(rng.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        rng.shuffle(candidates)
        
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            if not _is_move_valid(current_rows[ri], pi, new_val, static_pi, old_diffs, all_used):
                continue
                
            # Compute new metrics
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
                
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            new_sum_max = current_sum_max - old_row_max + cand_row_max if 'current_sum_max' in locals() else sum(row_maxes)
            row_maxes[ri] = old_row_max
            
            # Acceptance
            accept = False
            if new_global_max < best_val:
                accept = True
            elif new_global_max == best_val:
                if new_sum_max < best_sum_max:
                    accept = True
                elif new_sum_max == best_sum_max:
                    accept = rng.random() < 0.1
                else:
                    accept = rng.random() < 0.01
            else:
                delta = new_global_max - best_val
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    accept = rng.random() < prob
            
            if accept:
                # Update state
                new_diffs = _get_new_diffs_set(current_rows[ri], pi, new_val, static_pi)
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_sum_max = sum(row_maxes)
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if new_global_max < best_val or (new_global_max == best_val and current_sum_max < best_sum_max):
                    best_val = new_global_max
                    best_sum_max = current_sum_max
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        if no_improve > 12000:
            current_rows = [r[:] for r in best_rows]
            best_val = _global_max(current_rows)
            best_sum_max = sum(max(r) for r in best_rows)
            all_used = _compute_all_used(current_rows)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0

    return [r[:] for r in best_rows]

def _polish(rows, n_rows, k, deadline, rng):
    """
    Greedy polish: only accept improvements.
    """
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    current_sum_max = sum(max(r) for r in current_rows)
    all_used = _compute_all_used(current_rows)
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    best_sum_max = current_sum_max
    
    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == current_val]
    
    while time.time() < polish_deadline:
        if max_row_indices and rng.random() < 0.7:
            ri = rng.choice(max_row_indices)
        else:
            ri = rng.randrange(n_rows)
            
        if ri in max_row_indices and rng.random() < 0.5:
            pi = k
        else:
            pi = random.randrange(1, k + 1)
            
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(50, old_val - min_allowed)
        
        if max_decrease <= 0:
            continue
            
        candidates = [old_val - d for d in range(max_decrease, 0, -1)]
        static_pi = static_diffs[ri][pi]
        old_diffs = row_diffs[ri]
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            if not _is_move_valid(current_rows[ri], pi, new_val, static_pi, old_diffs, all_used):
                continue
                
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
                
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            new_sum_max = current_sum_max - old_row_max + cand_row_max
            row_maxes[ri] = old_row_max
            
            if new_global_max < current_val or (new_global_max == current_val and new_sum_max < current_sum_max):
                new_diffs = _get_new_diffs_set(current_rows[ri], pi, new_val, static_pi)
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                current_sum_max = new_sum_max
                
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
                if current_val < best_val or (current_val == best_val and current_sum_max < best_sum_max):
                    best_val = current_val
                    best_sum_max = current_sum_max
                    best_rows = [r[:] for r in current_rows]
                    max_row_indices = [i for i in range(n_rows) if row_maxes[i] == best_val]
                    
                moved = True
                break
    return [r[:] for r in best_rows]

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 108.0
    rng = random.Random(42)
    
    # 1. Multi-start initialization
    best_solution = _greedy_init(n_rows, k, rng)
    best_score = _global_max(best_solution)
    
    # Try a few random starts
    for start_i in range(3):
        if time.time() > deadline - 60:
            break
        rng_local = random.Random(42 + start_i * 1000)
        current = _greedy_init(n_rows, k, rng_local, randomize=True)
        if not current:
            continue
        curr_score = _global_max(current)
        if curr_score < best_score:
            best_score = curr_score
            best_solution = current
    
    # 2. Row-level SA
    if time.time() < deadline - 40:
        improved = _row_reconstruction_sa([r[:] for r in best_solution], n_rows, k, deadline, rng)
        if improved:
            curr_score = _global_max(improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
    
    # 3. Element-level SA
    if time.time() < deadline - 20:
        final = _element_sa([r[:] for r in best_solution], n_rows, k, deadline, rng)
        if final:
            final_score = _global_max(final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score
    
    # 4. Polish
    if time.time() < deadline - 5:
        final2 = _polish([r[:] for r in best_solution], n_rows, k, deadline, rng)
        if final2:
            final2_score = _global_max(final2)
            if final2_score < best_score:
                best_solution = final2
                best_score = final2_score
    
    return best_solution

# EVOLVE-BLOCK-END
