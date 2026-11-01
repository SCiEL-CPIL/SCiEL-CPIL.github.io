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

def _greedy_round_robin(n_rows, k, rng):
    """Round-robin greedy: extend rows one element at a time, choosing the row with smallest last element."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    
    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        best_r = -1
        best_last = -1
        order = list(range(n_rows))
        rng.shuffle(order)
        for r in order:
            if len(rows[r]) >= k + 1:
                continue
            last = rows[r][-1]
            if best_r == -1 or last < best_last:
                best_last = last
                best_r = r
        if best_r == -1:
            return None
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

def _try_greedy_variants(n_rows, k, rng, num_trials=30):
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
    
    # Also try round-robin
    for t in range(num_trials // 2):
        if _time_left() < 5.0:
            break
        rows = _greedy_round_robin(n_rows, k, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_max

def _backtracking_search(n_rows, k, upper_bound, rng, time_limit=20.0):
    """
    Backtracking search with forward checking to find a solution with max < upper_bound.
    """
    start_time = time.time()
    best_found = None
    best_max = upper_bound
    
    def search_with_order(rows, current_row, used_diffs, max_val_so_far, row_idx):
        nonlocal best_found, best_max
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            if max_val_so_far < best_max:
                best_max = max_val_so_far
                best_found = [list(r) for r in rows]
            return True
        
        if max_val_so_far >= best_max:
            return False
            
        if len(current_row) == 0:
            current_row = [0]
                
        if len(current_row) == k + 1:
            rows.append(list(current_row))
            success = search_with_order(rows, [], used_diffs, max_val_so_far, row_idx + 1)
            if success:
                return True
            rows.pop()
            return False
        
        last = current_row[-1]
        
        for x in range(last + 1, upper_bound + 1):
            valid = True
            new_diffs = []
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            current_row.append(x)
            used_diffs.update(new_diffs)
            new_max = max(max_val_so_far, x)
            
            if search_with_order(rows, current_row, used_diffs, new_max, row_idx):
                return True
            
            current_row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
                
        return False

    for _ in range(5):
        if time.time() - start_time > time_limit:
            break
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        search_with_order([], [], set(), 0, 0)
        
        if best_found is not None:
            break
            
    return best_found, best_max

def _local_search_reduce_max(rows, n_rows, k, rng, time_budget=30.0):
    """Try to reduce the maximum entry via local search with simulated annealing-like acceptance."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    # Temperature for simulated annealing
    temp = 10.0
    cooling = 0.99
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
        
        # Find rows with elements near the max
        max_val = max(max(rows[r]) for r in range(n_rows))
        target_rows = [r for r in range(n_rows) if max(rows[r]) >= max_val - 15]
        if not target_rows:
            target_rows = list(range(n_rows))
        r = rng.choice(target_rows)
        row = rows[r]
        
        # Try to reduce the last element
        old_val = row[-1]
        improved = False
        
        # First try deterministic reduction
        for new_val in range(old_val - 1, row[-2], -1):
            row_copy = row[:-1] + [new_val]
            nd = _compute_diffs_set(row_copy)
            if len(nd) != k*(k+1)//2:
                continue
            conflict = False
            for other_r in range(n_rows):
                if other_r == r:
                    continue
                if nd & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                row[-1] = new_val
                row_diffs[r] = nd
                all_diffs = set()
                for rr in range(n_rows):
                    all_diffs |= row_diffs[rr]
                cur_max = max(max(rows[rr]) for rr in range(n_rows))
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(rows[rr]) for rr in range(n_rows)]
                improved = True
                break
        
        if not improved:
            # Try a random perturbation with simulated annealing acceptance
            rr = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)
            old = rows[rr][pos]
            delta = rng.randint(-8, 8)
            new = old + delta
            if new <= rows[rr][pos-1] or (pos < k and new >= rows[rr][pos+1]) or new <= 0:
                continue
            
            row_copy = rows[rr][:]
            row_copy[pos] = new
            nd = _compute_diffs_set(row_copy)
            if len(nd) != k*(k+1)//2:
                continue
            
            # Check internal consistency
            if len(nd) != k*(k+1)//2:
                continue
            
            # Check against other rows
            conflict = False
            for other_r in range(n_rows):
                if other_r == rr:
                    continue
                if nd & row_diffs[other_r]:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Compute new max
            old_max = max(max(rows[x]) for x in range(n_rows))
            rows[rr][pos] = new
            new_max = max(max(rows[x]) for x in range(n_rows))
            
            # Accept if better or with probability exp(-delta/temp)
            if new_max < old_max:
                row_diffs[rr] = nd
                all_diffs = set()
                for x in range(n_rows):
                    all_diffs |= row_diffs[x]
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(rows[x]) for x in range(n_rows)]
            else:
                # Accept worse with probability
                prob = min(1.0, max(0.0, 1.0 - (new_max - old_max) / (temp * 5)))
                if rng.random() < prob:
                    row_diffs[rr] = nd
                    all_diffs = set()
                    for x in range(n_rows):
                        all_diffs |= row_diffs[x]
                else:
                    rows[rr][pos] = old  # revert
    
    return best_rows, best_max

def _multi_start_local_search(rows, n_rows, k, rng, time_budget=30.0, num_starts=5):
    """Run local search from multiple perturbed starting points."""
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    start = time.time()
    
    for s in range(num_starts):
        if time.time() - start > time_budget or _time_left() < 5.0:
            break
        
        # Create a perturbed copy
        perturbed = [list(r) for r in rows]
        # Apply small random perturbations to non-first elements
        for r in range(n_rows):
            for pos in range(1, k + 1):
                if rng.random() < 0.3:
                    delta = rng.randint(-3, 3)
                    new_val = perturbed[r][pos] + delta
                    if new_val > perturbed[r][pos-1] and (pos == k or new_val < perturbed[r][pos+1]):
                        perturbed[r][pos] = new_val
        
        # Check if perturbed is valid
        if not _validate_solution(perturbed, n_rows, k):
            continue
        
        cur_max = max(max(r) for r in perturbed)
        if cur_max > best_max + 10:  # Too bad, skip
            continue
        
        # Run local search from this starting point
        budget_for_this = (time_budget - (time.time() - start)) / (num_starts - s)
        ls_rows, ls_max = _local_search_reduce_max(perturbed, n_rows, k, rng, time_budget=budget_for_this)
        
        if ls_max < best_max:
            best_max = ls_max
            best_rows = [list(r) for r in ls_rows]
    
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Fast greedy to get a baseline
    best_rows, best_max = _try_greedy_variants(n_rows, k, rng, num_trials=25)
    
    if best_rows is None:
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
        
    # Phase 2: Backtracking search to improve
    time_for_bt = 25.0
    if _time_left() > time_for_bt + 15.0:
        bt_rows, bt_max = _backtracking_search(n_rows, k, best_max - 1, rng, time_limit=time_for_bt)
        if bt_rows is not None and bt_max < best_max:
            best_max = bt_max
            best_rows = bt_rows
            
    # Phase 3: Multi-start local search refinement
    if _time_left() > 15.0:
        ls_rows, ls_max = _multi_start_local_search(best_rows, n_rows, k, rng, 
                                                     time_budget=min(30.0, _time_left() - 10.0),
                                                     num_starts=5)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
    
    # Final single local search pass if time remains
    if _time_left() > 5.0:
        ls_rows, ls_max = _local_search_reduce_max(best_rows, n_rows, k, rng, 
                                                    time_budget=min(10.0, _time_left() - 3.0))
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
            
    # Final verification
    if not _validate_solution(best_rows, n_rows, k):
        # If invalid, try to fix by running greedy again
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
        
    return best_rows
