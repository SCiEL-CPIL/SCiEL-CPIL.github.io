"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random
import numpy as np

_START_TIME = None
_DEADLINE = 145.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_differences(row):
    """Compute all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_vectorized(n_rows, k, row_order=None, rng=None):
    """Fast greedy using sorted list of used differences."""
    if rng is None:
        rng = random.Random()
    if row_order is None:
        row_order = list(range(n_rows))
    
    used = set()
    rows = {}
    for r in row_order:
        rows[r] = [0]
    
    for r in row_order:
        row = rows[r]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Check if all new differences are unused
            new_diffs = [x - a for a in row]
            if not any(d in used for d in new_diffs):
                row.append(x)
                used.update(new_diffs)
        rows[r] = row
    
    result = [rows[r] for r in range(n_rows)]
    return result

def _greedy_constrained(n_rows, k, rng=None):
    """Round-robin greedy: extend the most constrained row each step."""
    if rng is None:
        rng = random.Random()
    
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    
    while placed < total_needed:
        if _time_left() < 2.0:
            break
        
        # Find the row that is most constrained (fewest valid next candidates)
        best_row = -1
        best_candidate = None
        best_score = float('inf')
        
        # Try rows in random order to break ties
        order = list(range(n_rows))
        rng.shuffle(order)
        
        for r in order:
            if len(rows[r]) >= k + 1:
                continue
            last = rows[r][-1]
            # Find the smallest valid next candidate
            x = last + 1
            while True:
                new_diffs = [x - a for a in rows[r]]
                if not any(d in used for d in new_diffs):
                    break
                x += 1
            # Score: prefer rows with smaller last element (more constrained)
            score = last
            if score < best_score:
                best_score = score
                best_row = r
                best_candidate = x
        
        if best_row == -1:
            break
        
        rows[best_row].append(best_candidate)
        for a in rows[best_row]:
            used.add(best_candidate - a)
        placed += 1
    
    for r in range(n_rows):
        if len(rows[r]) < k + 1:
            return None
    return rows

def _local_search_reduce_max(rows, n_rows, k, rng, max_iters=5000):
    """Try to reduce the maximum entry via local search.
    
    Strategy: find the row with the max element, try to move it to a smaller
    position. If conflicts arise, try to fix by adjusting other elements.
    """
    rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    
    if _time_left() < 3.0:
        return rows, best_max
    
    for iteration in range(max_iters):
        if _time_left() < 3.0:
            break
        
        # Find the row and position of the maximum element
        max_val = max(max(r) for r in rows)
        if max_val == best_max and rng.random() < 0.5:
            # Already at best, try a random move
            pass
        
        # Find all rows containing elements near the maximum
        target_rows = []
        for r in range(n_rows):
            if max(rows[r]) >= max_val - 5:
                target_rows.append(r)
        
        if not target_rows:
            break
        
        # Pick a random target row
        r = rng.choice(target_rows)
        row = rows[r]
        
        # Try to reduce the last element (which is likely the max)
        old_val = row[-1]
        
        # Try smaller values
        for new_val in range(old_val - 1, 0, -1):
            if new_val <= row[-2]:
                break
            # Check if new_val creates any conflicts
            new_diffs = set()
            valid = True
            for a in row[:-1]:
                d = new_val - a
                if d in _compute_differences(row[:-1]):
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                continue
            
            # Check against all other rows
            conflict = False
            for other_r in range(n_rows):
                if other_r == r:
                    continue
                other_diffs = _compute_differences(rows[other_r])
                if new_diffs & other_diffs:
                    conflict = True
                    break
            
            if not conflict:
                row[-1] = new_val
                cur_max = max(max(rr) for rr in rows)
                if cur_max < best_max:
                    best_max = cur_max
                break
    
    return rows, best_max

def _simulated_annealing(rows, n_rows, k, rng, time_budget=10.0):
    """Simulated annealing to reduce the maximum entry.
    
    State: the rows. Move: pick a random row and a random position,
    change the value to a nearby value. Accept if valid or with probability
    based on temperature.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    
    start_time = time.time()
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    while T > T_min and (time.time() - start_time) < time_budget:
        if _time_left() < 3.0:
            break
        
        # Pick a random row and position (not the first, which is 0)
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        row = rows[r]
        old_val = row[pos]
        
        # Propose a new value
        delta = rng.randint(-10, 10)
        new_val = old_val + delta
        
        # Must maintain increasing order
        if pos > 0 and new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
        
        # Compute old differences for this row
        old_diffs = set()
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                old_diffs.add(row[j] - row[i])
        
        # Compute new differences
        row[pos] = new_val
        new_diffs = set()
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                new_diffs.add(row[j] - row[i])
        
        # Check global validity
        valid = True
        for other_r in range(n_rows):
            if other_r == r:
                continue
            other_diffs = set()
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    other_diffs.add(rows[other_r][j] - rows[other_r][i])
            if new_diffs & other_diffs:
                valid = False
                break
        
        if valid:
            cur_max = max(max(rr) for rr in rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(rr) for rr in rows]
        else:
            # Revert
            row[pos] = old_val
            # Accept with probability based on temperature (penalize conflicts)
            # Count number of conflicts
            conflict_count = 0
            for other_r in range(n_rows):
                if other_r == r:
                    continue
                other_diffs = set()
                for i in range(k + 1):
                    for j in range(i + 1, k + 1):
                        other_diffs.add(rows[other_r][j] - rows[other_r][i])
                conflict_count += len(new_diffs & other_diffs)
            
            # Accept bad moves with low probability
            if rng.random() < 0.01 and conflict_count <= 2:
                row[pos] = new_val  # Keep the bad move
            else:
                row[pos] = old_val
        
        T *= cooling
    
    return best_rows, best_max

def _greedy_with_randomization(n_rows, k, rng, num_trials=20):
    """Try multiple greedy strategies with randomization."""
    best_rows = None
    best_max = float('inf')
    
    for trial in range(num_trials):
        if _time_left() < 5.0:
            break
        
        # Random row order
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        rows = _greedy_vectorized(n_rows, k, row_order=row_order, rng=rng)
        if rows is None:
            continue
        
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
    
    # Also try constrained greedy
    for trial in range(num_trials):
        if _time_left() < 5.0:
            break
        
        rows = _greedy_constrained(n_rows, k, rng=rng)
        if rows is None:
            continue
        
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = rows
    
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Fast greedy with randomization
    best_rows, best_max = _greedy_with_randomization(n_rows, k, rng, num_trials=30)
    
    if best_rows is None:
        # Fallback: simple sequential greedy
        best_rows = _greedy_vectorized(n_rows, k, rng=rng)
        best_max = max(max(r) for r in best_rows)
    
    # Phase 2: Local search to reduce max
    if _time_left() > 10.0:
        rows, lm_max = _local_search_reduce_max(best_rows, n_rows, k, rng, max_iters=20000)
        if lm_max < best_max:
            best_max = lm_max
            best_rows = rows
    
    # Phase 3: Simulated annealing if time permits
    if _time_left() > 15.0:
        sa_rows, sa_max = _simulated_annealing(best_rows, n_rows, k, rng, time_budget=_time_left() - 5.0)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Final verification
    all_diffs = set()
    for row in best_rows:
        row_diffs = _compute_differences(row)
        if row_diffs & all_diffs:
            # Invalid - should not happen, but return anyway
            pass
        all_diffs |= row_diffs
    
    return best_rows
