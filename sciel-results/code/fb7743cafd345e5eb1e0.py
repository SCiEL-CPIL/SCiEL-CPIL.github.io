import time
import random
import math
import sys

# EVOLVE-BLOCK-START

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

def _compute_new_diffs_fast(row, pi, new_val, static_diffs_pi):
    """
    Given a row with position pi replaced by new_val, compute the full diff set
    using the precomputed static diffs plus the k diffs involving pi.
    Returns (new_diffs_set, valid_internal).
    """
    n = len(row)
    new_diffs = set(static_diffs_pi)
    valid = True
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in new_diffs:
            valid = False
            break
        new_diffs.add(d)
    return new_diffs, valid

def _check_conflict_fast(new_diffs, old_diffs, all_used):
    """
    Check if new_diffs conflicts with all_used - old_diffs.
    Only need to check the k new diffs (those not in old_diffs).
    """
    for d in new_diffs:
        if d in all_used and d not in old_diffs:
            return True
    return False

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            valid = True
            for a in row:
                if (x - a) in used:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def _randomized_greedy_row(k, all_used, max_val_hint=200):
    """
    Generate a single row using randomized greedy.
    Returns (row, diffs_set) or (None, None) if fails.
    """
    row = [0]
    local_diffs = set()
    for step in range(k):
        candidates = []
        # Try values starting from min possible
        min_val = row[-1] + 1
        # We want to explore a bit, so we look up to min_val + 30
        max_search = min_val + 30
        for v in range(min_val, max_search):
            # Check against local_diffs and all_used
            ok = True
            for a in row:
                d = v - a
                if d in local_diffs or d in all_used:
                    ok = False
                    break
            if ok:
                candidates.append(v)
                
        if not candidates:
            return None, None
            
        # Pick randomly from the top few smallest candidates to balance exploration/exploitation
        if len(candidates) > 5:
            v = random.choice(candidates[:5])
        else:
            v = random.choice(candidates)
            
        row.append(v)
        for a in row[:-1]:
            local_diffs.add(v - a)
            
    return row, local_diffs

def _greedy_construct(n_rows, k, random_seed=0):
    """
    Greedy construction: build rows one by one, choosing smallest possible values.
    Uses randomization to explore different starting points.
    """
    rng = random.Random(random_seed)
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Try values starting from current_max + 1, with some randomization
            base = current_max + 1
            # Look at a window of values
            window = 20
            for v in range(base, base + window):
                valid = True
                for a in row:
                    d = v - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    # Check internal validity (no duplicate diffs in this row)
                    internal_ok = True
                    for a in row:
                        d = v - a
                        # This check is redundant if used_diffs only contains external diffs,
                        # but we must ensure no two diffs in the new row are the same.
                        # Since we build sequentially, we check against diffs already in 'row'
                        # that we haven't added to used_diffs yet? No, we add immediately.
                        # Wait, we haven't added the new diffs to used_diffs yet.
                        # So we need to check if d is already in the set of diffs formed by 'row' + v.
                        # The diffs formed by 'row' are already in used_diffs? No, only if we added them.
                        # In this greedy, we add to used_diffs immediately after choosing.
                        # So 'used_diffs' contains all diffs from previous rows AND the current row so far.
                        # Thus, checking 'd in used_diffs' is sufficient for both external and internal conflicts.
                        pass
                    row.append(v)
                    for a in row[:-1]:
                        used_diffs.add(v - a)
                    current_max = v
                    found = True
                    break
            if not found:
                # If we can't find a value in the window, expand window or fail
                # For robustness, just return None
                return None
        rows.append(row)
    return rows

def _local_search_minimize(rows, n_rows, k, deadline):
    """
    Local search that tries to reduce the maximum value by moving elements to smaller positions.
    Uses incremental diff tracking for speed.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    # Track which diffs are "hot" (involved in the max value)
    max_val = max(max(r) for r in best_rows)
    
    while time.time() < deadline - 3:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    diffs_involving.add(old_val - row[i])
                else:
                    diffs_involving.add(row[i] - old_val)
            
            forbidden_diffs = all_diffs - diffs_involving
            
            # Try to find smaller valid value
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                found = False
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
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        break
                
                if found:
                    continue
        
        if not progress:
            break
            
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Multi-start Initialization
    best_rows = None
    best_val = float('inf')
    
    # Try a few greedy starts
    for s in range(3):
        if time.time() > deadline - 30:
            break
        rows = _greedy_init(n_rows, k)
        val = _global_max(rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
            
    # Fallback
    if best_rows is None:
        best_rows = _greedy_init(n_rows, k)
        best_val = _global_max(best_rows)

    # ILS Loop
    while time.time() < deadline:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used = _compute_all_used(current_rows)
        row_maxes = [max(r) for r in current_rows]
        
        # Phase 1: Local Search (SA)
        T = 50.0
        T_min = 0.5
        cool = 0.9999
        
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k) for r in current_rows]
        
        steps_in_phase = 0
        max_steps_phase = 5000
        
        while time.time() < deadline and steps_in_phase < max_steps_phase:
            # Pick row and position
            row_weights = [row_maxes[r] for r in range(n_rows)]
            total_w = sum(row_weights)
            if total_w > 0:
                cum_w = []
                s = 0
                for w in row_weights:
                    s += w
                    cum_w.append(s)
                r_val = random.random() * s
                ri = 0
                while ri < n_rows - 1 and cum_w[ri] < r_val:
                    ri += 1
            else:
                ri = random.randrange(n_rows)
                
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(20, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
                
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
            max_increase = min(10, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
                
            if random.random() < 0.05:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 10
                if low <= high:
                    candidates.append(random.randint(low, high))
                    
            if not candidates:
                steps_in_phase += 1
                T = max(T_min, T * cool)
                continue
                
            random.shuffle(candidates)
            
            static_pi = static_diffs[ri][pi]
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                    
                new_diffs, valid_internal = _compute_new_diffs_fast(
                    current_rows[ri], pi, new_val, static_pi)
                    
                if not valid_internal:
                    continue
                    
                old_diffs = row_diffs[ri]
                
                if _check_conflict_fast(new_diffs, old_diffs, all_used):
                    continue
                    
                if pi == k:
                    cand_row_max = new_val
                else:
                    cand_row_max = current_rows[ri][k]
                    
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= old_diffs
                    all_used |= new_diffs
                    current_rows[ri][pi] = new_val
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = new_global_max
                    static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        
                    moved = True
                    break
            
            if not moved:
                pass
                
            T = max(T_min, T * cool)
            steps_in_phase += 1
            
        # Phase 2: Perturbation (Structural Change)
        if time.time() < deadline - 5:
            # Pick a random row to reconstruct
            ri = random.randrange(n_rows)
            
            # Remove row ri from all_used
            old_row = current_rows[ri]
            old_diffs = _row_diffs_set(old_row)
            all_used -= old_diffs
            
            # Reconstruct row ri using randomized greedy
            new_row, new_diffs = _randomized_greedy_row(k, all_used)
            
            if new_row is not None:
                # Add new diffs to all_used
                all_used |= new_diffs
                
                # Update current state
                current_rows[ri] = new_row
                new_row_max = max(new_row)
                row_maxes[ri] = new_row_max
                current_val = max(row_maxes)
                
                # If this perturbation immediately yields a better solution, update best
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
            else:
                # Failed to reconstruct, restore old row
                all_used |= old_diffs

    return best_rows

# EVOLVE-BLOCK-END
