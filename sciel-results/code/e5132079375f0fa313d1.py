"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_set(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate_solution(rows, n_rows, k):
    """Check that all differences across all rows are distinct and rows are valid."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k):
            if row[i] >= row[i+1]:
                return False
        d = _compute_diffs_set(row)
        if len(d) != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, k, row_order, rng, randomized=False, rand_prob=0.15, window=30):
    """Greedy: build rows in given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            if randomized and rng.random() < rand_prob:
                # Randomized: try a random candidate in a window
                candidates = []
                for candidate in range(x, x + window):
                    new_diffs = {candidate - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(candidate)
                if candidates:
                    x = rng.choice(candidates)
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    continue
            # Standard greedy
            while True:
                ok = True
                for a in row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows

def _try_greedy_variants(n_rows, k, rng, num_trials=30):
    """Try multiple greedy strategies and return best."""
    best_rows = None
    best_max = float('inf')
    
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        # Random row order with sequential build, some randomized
        order = list(range(n_rows))
        rng.shuffle(order)
        use_random = rng.random() < 0.5
        rows = _greedy_build(n_rows, k, order, rng, randomized=use_random)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_max

def _rebuild_row(row_idx, rows, n_rows, k, rng, target_max=None):
    """Rebuild a single row using greedy with the current used differences from other rows."""
    other_diffs = set()
    for r in range(n_rows):
        if r != row_idx:
            other_diffs |= _compute_diffs_set(rows[r])
    
    # Try to build the row with a smaller max if target_max is given
    best_row = None
    best_row_max = float('inf')
    
    for attempt in range(20):
        row = [0]
        x = 0
        skip = 0
        while len(row) < k + 1:
            x += 1
            if skip > 0:
                skip -= 1
                continue
            new_diffs = {x - a for a in row}
            if not (new_diffs & other_diffs):
                row.append(x)
                # Occasionally skip ahead to create gaps
                if rng.random() < 0.3:
                    skip = rng.randint(1, 3)
        
        cur_max = max(row)
        if target_max is not None and cur_max > target_max:
            continue
        if cur_max < best_row_max:
            best_row_max = cur_max
            best_row = list(row)
    
    return best_row, best_row_max

def _simulated_annealing(rows, n_rows, k, rng, time_budget=60.0):
    """
    Simulated annealing to reduce the maximum entry.
    Moves: 
    1. Move a single element to a smaller value
    2. Rebuild an entire row
    Acceptance: accept if better, or with probability exp(-delta/T) if worse
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    # Precompute difference sets
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    
    def get_other_diffs(exclude_r):
        """Return set of all differences except those in row exclude_r."""
        other = set()
        for r in range(n_rows):
            if r != exclude_r:
                other |= row_diffs[r]
        return other
    
    def is_valid_move(new_row, r):
        """Check if new_row is valid given other rows."""
        if len(new_row) != k + 1:
            return False
        if new_row[0] != 0:
            return False
        for i in range(k):
            if new_row[i] >= new_row[i+1]:
                return False
        d = _compute_diffs_set(new_row)
        if len(d) != k * (k + 1) // 2:
            return False
        other_diffs = get_other_diffs(r)
        if d & other_diffs:
            return False
        return True
    
    T = 50.0  # Initial temperature
    cooling_rate = 0.995
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        cur_max = max(max(rows[r]) for r in range(n_rows))
        
        # Update best
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.7:
            # Move a single element
            r = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)
            old_val = rows[r][pos]
            
            # Try to decrease the value
            lo = rows[r][pos - 1] + 1
            hi = old_val - 1
            if pos < k:
                hi = min(hi, rows[r][pos + 1] - 1)
            
            if lo > hi:
                continue
            
            # Try a random new value in the range
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
            new_row = rows[r][:]
            new_row[pos] = new_val
            
            if not is_valid_move(new_row, r):
                continue
            
            new_max = max(max(rows[rr]) for rr in range(n_rows) if rr != r)
            new_max = max(new_max, max(new_row))
            delta = new_max - cur_max
            
            # Accept if better, or with probability exp(-delta/T)
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                rows[r] = new_row
                row_diffs[r] = _compute_diffs_set(new_row)
        
        else:
            # Rebuild a row
            r = rng.randrange(n_rows)
            # Target a smaller max
            target = max(rows[r]) - rng.randint(1, 5)
            new_row, new_row_max = _rebuild_row(r, rows, n_rows, k, rng, target_max=target)
            
            if new_row is None:
                continue
            
            if not is_valid_move(new_row, r):
                continue
            
            new_max = max(max(rows[rr]) for rr in range(n_rows) if rr != r)
            new_max = max(new_max, max(new_row))
            delta = new_max - cur_max
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                rows[r] = new_row
                row_diffs[r] = _compute_diffs_set(new_row)
        
        # Cool down
        T *= cooling_rate
        if T < 1.0:
            T = 1.0
    
    return best_rows, best_max

def _targeted_descent(rows, n_rows, k, rng, time_budget=30.0):
    """
    Structural optimization: Target the global maximum and try to reduce it.
    This is a deterministic descent with some randomization for tie-breaking or restarts.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    # Precompute difference sets for each row
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find the row with the current global maximum
        max_val = max(max(rows[r]) for r in range(n_rows))
        if max_val < best_max:
            best_max = max_val
            best_rows = [list(r) for r in rows]
        
        # Identify rows containing this max value
        max_rows = [r for r in range(n_rows) if max(rows[r]) == max_val]
        
        # Try to reduce the last element of one of these rows
        improved = False
        for r in max_rows:
            if _time_left() < 3.0:
                break
            row = rows[r]
            last_idx = len(row) - 1
            old_val = row[last_idx]
            
            # Try values from old_val - 1 down to row[last_idx - 1] + 1
            for new_val in range(old_val - 1, row[last_idx - 1], -1):
                base_diffs = _compute_diffs_set(row[:-1])
                
                # Check if new_val - a is in base_diffs
                valid_internal = True
                for a in row[:-1]:
                    d = new_val - a
                    if d in base_diffs:
                        valid_internal = False
                        break
                
                if not valid_internal:
                    continue
                
                # Check against other rows
                other_diffs = set()
                for other_r in range(n_rows):
                    if other_r != r:
                        other_diffs |= row_diffs[other_r]
                
                valid_external = True
                for a in row[:-1]:
                    d = new_val - a
                    if d in other_diffs:
                        valid_external = False
                        break
                
                if not valid_external:
                    continue
                
                # Valid move found!
                row[last_idx] = new_val
                row_diffs[r] = _compute_diffs_set(row)
                improved = True
                break
            
            if improved:
                break
        
        if not improved:
            # If we couldn't reduce the max, try to reduce the second max
            non_max_rows = [r for r in range(n_rows) if max(rows[r]) < max_val]
            if non_max_rows:
                r = rng.choice(non_max_rows)
                row = rows[r]
                last_idx = len(row) - 1
                old_val = row[last_idx]
                for new_val in range(old_val - 1, row[last_idx - 1], -1):
                    base_diffs = _compute_diffs_set(row[:-1])
                    
                    valid_internal = True
                    for a in row[:-1]:
                        d = new_val - a
                        if d in base_diffs:
                            valid_internal = False
                            break
                    
                    if not valid_internal:
                        continue
                    
                    other_diffs = set()
                    for other_r in range(n_rows):
                        if other_r != r:
                            other_diffs |= row_diffs[other_r]
                    
                    valid_external = True
                    for a in row[:-1]:
                        d = new_val - a
                        if d in other_diffs:
                            valid_external = False
                            break
                    
                    if not valid_external:
                        continue
                    
                    row[last_idx] = new_val
                    row_diffs[r] = _compute_diffs_set(row)
                    improved = True
                    break
            
            if not improved:
                break

    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Fast greedy to get a baseline
    best_rows, best_max = _try_greedy_variants(n_rows, k, rng, num_trials=30)
    
    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
        
    # Phase 2: Simulated Annealing (main improvement engine)
    if _time_left() > 15.0:
        sa_budget = min(60.0, _time_left() - 10.0)
        sa_rows, sa_max = _simulated_annealing(best_rows, n_rows, k, rng, time_budget=sa_budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Phase 3: Targeted Descent for final polish
    if _time_left() > 10.0:
        td_budget = min(30.0, _time_left() - 5.0)
        td_rows, td_max = _targeted_descent(best_rows, n_rows, k, rng, time_budget=td_budget)
        if td_max < best_max:
            best_max = td_max
            best_rows = td_rows

    # Phase 4: If time remains, try another round with a different seed
    if _time_left() > 20.0:
        rng2 = random.Random(123)
        alt_rows, alt_max = _try_greedy_variants(n_rows, k, rng2, num_trials=15)
        if alt_rows is not None:
            sa_budget2 = min(30.0, _time_left() - 10.0)
            alt_sa_rows, alt_sa_max = _simulated_annealing(alt_rows, n_rows, k, rng2, time_budget=sa_budget2)
            if alt_sa_max < best_max:
                best_max = alt_sa_max
                best_rows = alt_sa_rows
            if _time_left() > 10.0:
                td_budget2 = min(15.0, _time_left() - 5.0)
                alt_td_rows, alt_td_max = _targeted_descent(best_rows, n_rows, k, rng2, time_budget=td_budget2)
                if alt_td_max < best_max:
                    best_max = alt_td_max
                    best_rows = alt_td_rows

    # Final verification
    if not _validate_solution(best_rows, n_rows, k):
        # If invalid, try to fix by running greedy again
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
        
    return best_rows
