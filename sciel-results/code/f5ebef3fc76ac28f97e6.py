"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random
import sys
from collections import defaultdict

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

def _greedy_build(n_rows, k, row_order, rng):
    """Greedy: build rows in given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
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

def _greedy_interleaved(n_rows, k, rng, strategy="smallest"):
    """
    Interleaved greedy: at each step, pick a row to extend.
    strategy: 'smallest' (smallest last element), 'largest' (largest last element), 
              'random', 'balanced' (most elements placed)
    """
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    
    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        
        # Find candidate rows that need more elements
        candidates = [r for r in range(n_rows) if len(rows[r]) < k + 1]
        if not candidates:
            break
        
        if strategy == "smallest":
            best_r = min(candidates, key=lambda r: rows[r][-1])
        elif strategy == "largest":
            best_r = max(candidates, key=lambda r: rows[r][-1])
        elif strategy == "random":
            best_r = rng.choice(candidates)
        elif strategy == "balanced":
            best_r = max(candidates, key=lambda r: len(rows[r]))
        else:
            best_r = min(candidates, key=lambda r: rows[r][-1])
        
        row = rows[best_r]
        x = row[-1] + 1
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
                placed += 1
                break
            x += 1
    return rows

def _try_greedy_variants(n_rows, k, rng, num_trials=40):
    """Try multiple greedy strategies and return best."""
    best_rows = None
    best_max = float('inf')
    
    strategies = ["smallest", "largest", "random", "balanced"]
    
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        # Random row order with sequential build
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    # Try interleaved strategies
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        strategy = strategies[t % len(strategies)]
        rows = _greedy_interleaved(n_rows, k, rng, strategy=strategy)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_max

def _targeted_descent(rows, n_rows, k, rng, time_budget=60.0):
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
    
    def get_global_diffs_except_row(exclude_r):
        """Return set of all differences except those in row exclude_r."""
        all_d = set()
        for r in range(n_rows):
            if r == exclude_r:
                continue
            all_d |= row_diffs[r]
        return all_d

    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find the row with the current global maximum
        max_val = max(max(rows[r]) for r in range(n_rows))
        if max_val < best_max:
            best_max = max_val
            best_rows = [list(r) for r in rows]
            if best_max == 0: break # Theoretical minimum, though impossible for n>1, k>0
        
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
            # We want to keep it increasing
            for new_val in range(old_val - 1, row[last_idx - 1], -1):
                # Construct the hypothetical new row
                # The new differences involving the new last element are {new_val - a for a in row[:-1]}
                # The differences NOT involving the last element are the differences of row[:-1]
                
                # 1. Check internal consistency of the new row
                # The set of diffs for row[:-1] is fixed. Let's call it internal_base_diffs
                # We need to ensure new diffs don't conflict with internal_base_diffs
                # And new diffs are distinct among themselves (they are, since a's are distinct)
                
                # Efficient check:
                # new_diffs = {new_val - a for a in row[:-1]}
                # base_diffs = _compute_diffs_set(row[:-1])
                
                # To avoid recomputing base_diffs every time, we can precompute it for the prefix
                # But row[:-1] changes if we reduce previous elements. Here we only reduce the last one.
                # So base_diffs is constant for this row as long as we only touch the last element.
                
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
                
                # 2. Check against other rows
                # We need to ensure new_diffs do not intersect with diffs of any other row.
                # We can precompute the union of all diffs from other rows.
                other_diffs = set()
                for other_r in range(n_rows):
                    if other_r == r:
                        continue
                    other_diffs |= row_diffs[other_r]
                
                # Check new diffs against other_diffs
                valid_external = True
                for a in row[:-1]:
                    d = new_val - a
                    if d in other_diffs:
                        valid_external = False
                        break
                
                if not valid_external:
                    continue
                
                # Valid move found!
                # Apply it
                row[last_idx] = new_val
                row_diffs[r] = _compute_diffs_set(row)
                improved = True
                break
            
            if improved:
                break
        
        if not improved:
            # If we couldn't reduce the max, try to reduce the second max, or perform a small perturbation
            # to escape the local minimum.
            # Strategy: Pick a random row, pick a random element (not first), try to decrease it slightly.
            r = rng.randrange(n_rows)
            row = rows[r]
            if len(row) > 1:
                pos = rng.randrange(1, len(row))
                old_val = row[pos]
                # Try to decrease by 1 or 2
                for delta in [1, 2]:
                    new_val = old_val - delta
                    if new_val <= row[pos-1]:
                        continue
                    if pos < len(row)-1 and new_val >= row[pos+1]:
                        continue
                    
                    # Check validity
                    # This is more expensive as it affects diffs with both neighbors
                    # Construct new row
                    new_row = row[:]
                    new_row[pos] = new_val
                    new_diffs = _compute_diffs_set(new_row)
                    
                    if len(new_diffs) != k*(k+1)//2:
                        continue
                    
                    # Check against other rows
                    other_diffs = set()
                    for other_r in range(n_rows):
                        if other_r == r:
                            continue
                        other_diffs |= row_diffs[other_r]
                    
                    if new_diffs & other_diffs:
                        continue
                    
                    # Valid
                    rows[r] = new_row
                    row_diffs[r] = new_diffs
                    improved = True
                    break
            
            if not improved:
                # If nothing worked, we are stuck in a local minimum.
                # We could try a "kick": randomly change a value and accept if it doesn't increase max,
                # or use a simple restart if time allows.
                # For now, just break out of this iteration to avoid infinite loop on same state.
                # Or, try to swap the last element of two rows if it helps.
                # Let's try a simple random restart of the local search phase if we are stuck for a while.
                # But since we are in a loop, let's just continue to the next iteration, 
                # hoping the max_val check at the top triggers a best update if we accidentally improved.
                # Actually, if no improvement, we should probably break to avoid wasting time.
                # However, the "stuck" state might be resolved by a different choice in max_rows if there were ties.
                # Since we iterated through all max_rows, we are truly stuck on reducing the current global max.
                # We must change the structure. Let's try to reduce the max of a *different* row 
                # even if it's not the global max, to potentially open up space.
                
                # Pick a random row that is NOT the one with the global max (if possible)
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
                            if other_r == r:
                                continue
                            other_diffs |= row_diffs[other_r]
                        
                        valid_external = True
                        for a in row[:-1]:
                            d = new_val - a
                            if d in other_diffs:
                                valid_external = False
                                break
                        
                        if not valid_external:
                            continue
                        
                        # Apply
                        row[last_idx] = new_val
                        row_diffs[r] = _compute_diffs_set(row)
                        improved = True
                        break
                
                if not improved:
                    # Truly stuck. To avoid infinite loop, we can break out of the while loop 
                    # or introduce a random "jump". 
                    # A random jump: pick a random row, random pos, random delta.
                    # If valid, accept even if max doesn't decrease immediately (it might enable future decreases).
                    # But for simplicity and safety, let's just break if we can't make any progress.
                    # The outer loop in construct() will handle restarting if needed.
                    break

    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Fast greedy to get a baseline
    # Use more trials to find a good starting point
    best_rows, best_max = _try_greedy_variants(n_rows, k, rng, num_trials=50)
    
    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
        
    # Phase 2: Targeted Descent
    # This is the main improvement engine. It focuses on reducing the global maximum.
    if _time_left() > 10.0:
        ls_rows, ls_max = _targeted_descent(best_rows, n_rows, k, rng, time_budget=_time_left() - 5.0)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Phase 3: If time remains, try another round of greedy + descent with a different seed
    if _time_left() > 20.0:
        rng2 = random.Random(123)
        alt_rows, alt_max = _try_greedy_variants(n_rows, k, rng2, num_trials=20)
        if alt_rows is not None:
            # Refine the alternative
            alt_refined_rows, alt_refined_max = _targeted_descent(alt_rows, n_rows, k, rng2, time_budget=min(15.0, _time_left() - 5.0))
            if alt_refined_max < best_max:
                best_max = alt_refined_max
                best_rows = alt_refined_rows

    # Final verification
    if not _validate_solution(best_rows, n_rows, k):
        # If invalid, try to fix by running greedy again
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
        
    return best_rows
