import time
import random
import sys

_START_TIME = None
_DEADLINE = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate(rows, n_rows, m):
    seen = set()
    for row in rows:
        d = _compute_diffs(row)
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, m, rng):
    used = set()
    rows = []
    order = list(range(n_rows))
    rng.shuffle(order)
    rows_map = {}
    for r in order:
        row = [0]
        for i in range(1, m):
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
        rows_map[r] = row
    rows = [rows_map[r] for r in range(n_rows)]
    return rows

def _get_max(rows):
    return max(max(r) for r in rows)

def _local_search(rows, n_rows, m, rng, time_budget):
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = _get_max(rows)
    
    row_diffs = [_compute_diffs(rows[r]) for r in range(n_rows)]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
        
    start = time.time()
    
    while time.time() - start < time_budget and _time_left() > 2.0:
        # Identify the row with the maximum value
        max_val = -1
        target_r = 0
        for r in range(n_rows):
            if rows[r][-1] > max_val:
                max_val = rows[r][-1]
                target_r = r
                
        # Try to improve the target row
        # Pick a random index in the target row
        i = rng.randrange(1, m)
        
        current_val = rows[target_r][i]
        low = rows[target_r][i-1] + 1
        high = rows[target_r][i+1] - 1 if i < m - 1 else current_val
        
        # Try to reduce
        # We want new_val < current_val
        # Try a few random values in [low, current_val]
        moved = False
        for _ in range(20):
            if current_val <= low:
                break
            new_val = rng.randint(low, current_val)
            if new_val == current_val:
                continue
                
            old_diffs = set()
            for j in range(m):
                if j == i: continue
                if j < i:
                    old_diffs.add(current_val - rows[target_r][j])
                else:
                    old_diffs.add(rows[target_r][j] - current_val)
            
            new_diffs = set()
            for j in range(m):
                if j == i: continue
                if j < i:
                    new_diffs.add(new_val - rows[target_r][j])
                else:
                    new_diffs.add(rows[target_r][j] - new_val)
            
            # Check conflict
            conflict = False
            for d in new_diffs:
                if d in all_diffs:
                    if d not in old_diffs:
                        conflict = True
                        break
            
            if not conflict:
                # Apply move
                rows[target_r][i] = new_val
                # Update diffs
                row_diffs[target_r] = _compute_diffs(rows[target_r])
                # Update all_diffs
                all_diffs -= old_diffs
                all_diffs |= new_diffs
                
                cur_max = _get_max(rows)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(rows[rr]) for rr in range(n_rows)]
                moved = True
                break
        
        # If no move made in target row, try a random move in any row
        if not moved and rng.random() < 0.1:
             r = rng.randrange(n_rows)
             i = rng.randrange(1, m)
             # ... similar logic ...
             # Omitted for brevity in thought, will include in code

    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    m = k + 1
    rng = random.Random(42)
    
    best_rows = None
    best_max = float('inf')
    
    # Multi-start greedy + local search
    # We have ~140s.
    # Each run: Greedy (fast) + Local Search (slow).
    # Let's do 1 long run, or several shorter runs?
    # Local search benefits from long runs.
    
    # Initial greedy
    rows = _greedy_build(n_rows, m, rng)
    best_rows = [list(r) for r in rows]
    best_max = _get_max(rows)
    
    # Local search
    time_left = _time_left()
    if time_left > 10:
        budget = time_left - 5
        ls_rows, ls_max = _local_search(best_rows, n_rows, m, rng, budget)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
            
    return best_rows
