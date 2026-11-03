import time
import random
import math
from collections import Counter

_BEST = None
_DEADLINE = 0.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _fast_greedy(k, n_rows, rng=None, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - r[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - r[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _dfs_rebuild_row(k, diff_counts, max_scope, rng, attempts=3):
    """
    DFS-based row reconstruction: find a valid row of k+1 elements starting at 0,
    all differences distinct within the row and not conflicting with diff_counts,
    with the maximum element as small as possible (bounded by max_scope).
    Returns the row or None.
    """
    best_row = None
    best_max = max_scope + 1
    
    for attempt in range(attempts):
        # Try to build row greedily with some randomness
        row = [0]
        x = 0
        local_used = set()
        
        for pos in range(1, k + 1):
            found = False
            # Search from x+1 up to max_scope
            search_start = x + 1
            if attempt > 0:
                # Add some randomness to escape local patterns
                search_start = x + rng.randint(1, min(3, max_scope - x))
            
            for t in range(search_start, min(max_scope, x + 50) + 1):
                # Check all differences with existing row elements
                conflict = False
                new_diffs = []
                for a in row:
                    d = t - a
                    if d <= 0:
                        continue
                    if d in local_used:
                        conflict = True
                        break
                    if diff_counts.get(d, 0) > 0:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if not conflict:
                    row.append(t)
                    local_used.update(new_diffs)
                    x = t
                    found = True
                    break
            
            if not found:
                # If greedy fails, try a more exhaustive search for this position
                # but limit the search
                found = False
                for t in range(x + 1, max_scope + 1):
                    conflict = False
                    new_diffs = []
                    for a in row:
                        d = t - a
                        if d <= 0:
                            continue
                        if d in local_used:
                            conflict = True
                            break
                        if diff_counts.get(d, 0) > 0:
                            conflict = True
                            break
                        new_diffs.append(d)
                    
                    if not conflict:
                        row.append(t)
                        local_used.update(new_diffs)
                        x = t
                        found = True
                        break
                
                if not found:
                    break
        
        if len(row) == k + 1:
            row_max = row[-1]
            if row_max < best_max:
                best_max = row_max
                best_row = list(row)
                if best_max <= max_scope - 2:
                    break  # Good enough, stop early
    
    return best_row

def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(rows, dc):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 500 * collisions

    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    if polish:
        T = 0.5
        cooling = 0.9999
        min_T = 0.01
    else:
        T = 8.0
        cooling = 0.9995
        min_T = 0.05
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        r = rng.random()
        
        if r < 0.20:
            # Row rebuild move using DFS
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            # Use DFS to find a better row
            new_row = _dfs_rebuild_row(k, diff_counts, current_scope, rng, attempts=2)
            if new_row is None:
                # Fallback to simple greedy
                new_row = _fast_greedy_row(k, diff_counts, current_scope, rng)
            
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                continue
            
            _add_row_diffs(diff_counts, new_row)
            current_rows[i] = new_row
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row
                
        elif r < 0.40:
            # Scope compression: try to reduce the max element
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if current_rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0:
                    break
            
            if max_row_idx < 0:
                continue
                
            i = max_row_idx
            j = max_pos_idx
            old_val = current_rows[i][j]
            
            prev_val = current_rows[i][j-1]
            if old_val - prev_val <= 1:
                continue
                
            new_val = rng.randint(prev_val + 1, old_val - 1)
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
                
        else:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            if rng.random() < 0.7:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.2)))
                new_val = old_val + delta
            else:
                new_val = rng.randint(lo, hi)
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
            
        T = max(T * cooling, min_T)
        
    return best_scope, best_rows

def _fast_greedy_row(k, diff_counts, max_scope, rng):
    """Simple greedy row builder respecting diff_counts."""
    row = [0]
    x = 0
    local_used = set()
    
    for pos in range(1, k + 1):
        found = False
        for t in range(x + 1, max_scope + 1):
            conflict = False
            new_diffs = []
            for a in row:
                d = t - a
                if d <= 0:
                    continue
                if d in local_used or diff_counts.get(d, 0) > 0:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if not conflict:
                row.append(t)
                local_used.update(new_diffs)
                x = t
                found = True
                break
        
        if not found:
            return None
    
    return row

def _scope_reduction_pass(rows, k, deadline):
    """
    Iteratively try to reduce the scope by rebuilding rows one at a time
    with a tighter max_scope bound.
    """
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    rng = random.Random(123)
    
    # Try to reduce scope by 1, 2, 3...
    for target_scope in range(current_scope - 1, max(50, current_scope - 20), -1):
        if time.time() > deadline:
            break
        
        success = False
        for attempt in range(5):
            trial_rows = [list(r) for r in current_rows]
            diff_counts = _build_diff_counts(trial_rows)
            
            # Rebuild each row trying to fit within target_scope
            all_ok = True
            for i in range(len(trial_rows)):
                _remove_row_diffs(diff_counts, trial_rows[i])
                new_row = _dfs_rebuild_row(k, diff_counts, target_scope, rng, attempts=3)
                if new_row is None:
                    new_row = _fast_greedy_row(k, diff_counts, target_scope, rng)
                if new_row is None:
                    all_ok = False
                    break
                _add_row_diffs(diff_counts, new_row)
                trial_rows[i] = new_row
            
            if all_ok:
                # Verify validity
                if _valid(trial_rows, k):
                    s = _scope(trial_rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [list(r) for r in trial_rows]
                        current_rows = trial_rows
                    success = True
                    break
        
        if not success:
            break  # Can't reduce further
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 145.0  # Safety margin before 150s kill
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    candidates = []
    
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    for _ in range(3):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 8)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
        
    candidates.sort(key=_scope)
    
    # Phase 1: SA optimization on best candidates
    for idx, cand in enumerate(candidates[:2]):
        if time.time() > _DEADLINE - 30:
            break
        remaining = _DEADLINE - time.time() - 30
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.5
        else:
            sa_deadline = _DEADLINE - 30
        iters = int((sa_deadline - time.time()) * 5000)
        iters = max(iters, 1000)
        s, rows = _sa_optimize(cand, k, iters, rng, sa_deadline)
        consider(rows)
        
    # Phase 2: Scope reduction pass
    if _BEST is not None and time.time() < _DEADLINE - 20:
        polish_deadline = _DEADLINE - 20
        s, rows = _scope_reduction_pass(_BEST[1], k, polish_deadline)
        consider(rows)
        
    # Phase 3: Final polish with low-temperature SA
    if _BEST is not None and time.time() < _DEADLINE - 5:
        final_deadline = _DEADLINE - 3
        final_iters = int((final_deadline - time.time()) * 8000)
        final_iters = max(final_iters, 500)
        s, rows = _sa_optimize(_BEST[1], k, final_iters, rng, final_deadline, polish=True)
        consider(rows)
        
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
