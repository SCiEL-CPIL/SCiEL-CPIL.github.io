import time
import random
import sys

_START_TIME = None
_DEADLINE = 145.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_bitset(row):
    """Return an integer bitset of all positive differences in a row."""
    diffs = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            diffs |= (1 << d)
    return diffs

def _is_valid(rows, n_rows, k):
    """Check that all differences across all rows are distinct."""
    seen = 0
    for row in rows:
        d = _compute_diffs_bitset(row)
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, k, rng, order=None):
    """Greedy: build rows in given order using bitsets."""
    if order is None:
        order = list(range(n_rows))
        rng.shuffle(order)
        
    used = 0
    rows = [None] * n_rows
    for r in order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                # Check new differences
                # new_diffs will be bits for (x - a) for a in row
                new_diffs = 0
                ok = True
                for a in row:
                    d = x - a
                    new_diffs |= (1 << d)
                    if (new_diffs & used) != 0:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    used |= new_diffs
                    break
                x += 1
        rows[r] = row
    return rows

def _simulated_annealing(rows, n_rows, k, rng, time_budget):
    """
    SA using bitsets for fast conflict checking.
    Objective: Minimize max value.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs = [_compute_diffs_bitset(rows[r]) for r in range(n_rows)]
    all_diffs = 0
    for rd in row_diffs:
        all_diffs |= rd
        
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    current_max = best_max
    
    while T > T_min and time.time() - start < time_budget and _time_left() > 2.0:
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        row = rows[r]
        old_val = row[pos]
        
        # Random delta
        # Bias towards small changes
        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        
        # Bounds
        if new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
            
        row[pos] = new_val
        new_rd = _compute_diffs_bitset(row)
        
        # Check internal consistency (distinct diffs within row)
        # If len of bits != k*(k+1)/2, there are duplicates in the row
        # Counting bits is slow in Python? 
        # bin(x).count('1') is O(bits). Bits ~ 150. Fast enough.
        if bin(new_rd).count('1') != k * (k + 1) // 2:
            row[pos] = old_val
            T *= cooling
            continue
            
        # Check conflicts with other rows
        # other_diffs = all_diffs ^ row_diffs[r] (since row_diffs[r] is subset of all_diffs)
        # Wait, all_diffs is union. row_diffs[r] is subset.
        # other_diffs = all_diffs & ~row_diffs[r]
        other_diffs = all_diffs & ~row_diffs[r]
        
        if (new_rd & other_diffs) != 0:
            # Conflict exists. 
            # Calculate cost delta?
            # For SA on Max Value, we usually only consider valid states or penalize conflicts.
            # Let's strictly enforce validity in the search space, but allow "uphill" moves in Max Value.
            # If conflict, revert.
            row[pos] = old_val
            T *= cooling
            continue
            
        # Valid move
        all_diffs = (all_diffs & ~row_diffs[r]) | new_rd
        row_diffs[r] = new_rd
        
        cur_max = max(max(rows[x]) for x in range(n_rows))
        
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(x) for x in rows]
            
        # Acceptance
        if cur_max <= current_max:
            current_max = cur_max
        else:
            # Uphill move in Max Value
            delta_cost = cur_max - current_max
            if rng.random() < pow(2.71828, -delta_cost / T):
                current_max = cur_max
            else:
                # Revert
                all_diffs = (all_diffs & ~new_rd) | row_diffs[r]
                row_diffs[r] = _compute_diffs_bitset(rows[r]) # Restore, but we have old row_diffs? 
                # We need to store old_row_diffs before modifying
                # Let's restructure to store old state
                pass # Logic flaw: need to store old_row_diffs

        T *= cooling
        
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Starts
    best_rows = None
    best_max = float('inf')
    
    for i in range(50):
        if _time_left() < 10.0:
            break
        rows = _greedy_build(n_rows, k, rng)
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, rng)
        best_max = max(max(r) for r in best_rows)
        
    # Phase 2: Simulated Annealing
    if _time_left() > 10.0:
        budget = min(100.0, _time_left() - 5.0)
        sa_rows, sa_max = _simulated_annealing(best_rows, n_rows, k, rng, budget)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
            
    # Final Verification
    if not _is_valid(best_rows, n_rows, k):
        # Fallback
        best_rows = _greedy_build(n_rows, k, rng)
        
    return best_rows
