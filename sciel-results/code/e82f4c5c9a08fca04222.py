import time
import random
import math
import numpy as np

_DEADLINE = 0.0
_BEST = None

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

def _fast_greedy(k, n_rows, rng, spread=False):
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
    """Build a dictionary mapping each difference value to its count across all rows."""
    counts = {}
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] = counts.get(d, 0) + 1
    return counts

def _remove_row_diffs(counts, row):
    """Remove all differences from a single row from the counts."""
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            counts[d] -= 1
            if counts[d] == 0:
                del counts[d]

def _add_row_diffs(counts, row):
    """Add all differences from a single row to the counts."""
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            counts[d] = counts.get(d, 0) + 1

def _row_rebuild(i, rows, counts, k, rng, max_search=500):
    """
    Rebuild row i using greedy construction while respecting all other rows' differences.
    Returns the new row or None if it fails.
    """
    # Get all differences used by other rows
    other_diffs = set()
    for ri in range(len(rows)):
        if ri == i:
            continue
        for a in range(len(rows[ri])):
            for b in range(a + 1, len(rows[ri])):
                other_diffs.add(rows[ri][b] - rows[ri][a])
    
    # Greedily construct new row
    new_row = [0]
    x = 0
    for _pos in range(1, k + 1):
        found = False
        for t in range(x + 1, x + max_search):
            conflict = False
            for a in new_row:
                d = t - a
                if d in other_diffs:
                    conflict = True
                    break
                if d <= 0:
                    continue
            if not conflict:
                # Also check that new differences don't conflict with each other
                # (they can't since we're adding in order)
                new_row.append(t)
                for a in new_row[:-1]:
                    other_diffs.add(t - a)
                x = t
                found = True
                break
        if not found:
            return None
    
    return new_row

def _sa_optimize(rows, k, n_rows, deadline, rng, iters_cap=500000):
    """
    Simulated annealing with incremental diff counting.
    """
    global _BEST
    
    diff_counts = _build_diff_counts(rows)
    n_diffs_per_row = k * (k + 1) // 2
    total_expected_diffs = n_rows * n_diffs_per_row
    
    def is_valid_counts(counts):
        """Check if all counts are exactly 1 (valid solution)."""
        if len(counts) != total_expected_diffs:
            return False
        for v in counts.values():
            if v != 1:
                return False
        return True
    
    def get_collision_count(counts):
        """Number of differences with count > 1."""
        return sum(1 for v in counts.values() if v > 1)
    
    current_scope = _scope(rows)
    current_collisions = get_collision_count(diff_counts)
    
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    best_counts = dict(diff_counts)
    
    if current_collisions == 0:
        if _BEST is None or current_scope < _BEST[0]:
            _BEST = (current_scope, best_rows)
    
    T = 10.0
    cooling = 0.9998
    min_T = 0.1
    
    for it in range(iters_cap):
        if time.time() > deadline:
            break
        
        r = rng.random()
        
        if r < 0.40:
            # Single element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = rows[i][j]
            prev_val = rows[i][j - 1]
            next_val = rows[i][j + 1] if j < k else rows[i][k] + 1000
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            span = hi - lo
            delta = int(rng.gauss(0, max(1, span * 0.3)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
            
            if new_val == old_val:
                continue
            
            # Remove old row's diffs
            _remove_row_diffs(diff_counts, rows[i])
            
            # Apply change
            rows[i][j] = new_val
            
            # Add new row's diffs
            _add_row_diffs(diff_counts, rows[i])
            
            new_scope = max(max(r) for r in rows)
            new_collisions = get_collision_count(diff_counts)
            
            # Cost: collisions first, then scope
            old_cost = current_collisions * 10000 + current_scope
            new_cost = new_collisions * 10000 + new_scope
            
            if new_cost < old_cost or rng.random() < math.exp(-(new_cost - old_cost) / max(T, min_T)):
                current_scope = new_scope
                current_collisions = new_collisions
                if new_collisions == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                    best_counts = dict(diff_counts)
                    if _BEST is None or best_scope < _BEST[0]:
                        _BEST = (best_scope, best_rows)
            else:
                # Revert
                _remove_row_diffs(diff_counts, rows[i])
                rows[i][j] = old_val
                _add_row_diffs(diff_counts, rows[i])
                
        elif r < 0.60:
            # Row rebuild
            i = rng.randint(0, n_rows - 1)
            old_row = list(rows[i])
            
            # Remove old row's diffs
            _remove_row_diffs(diff_counts, rows[i])
            
            # Rebuild
            new_row = _row_rebuild(i, rows, diff_counts, k, rng, max_search=300)
            
            if new_row is not None:
                rows[i] = new_row
                _add_row_diffs(diff_counts, new_row)
                
                new_scope = max(max(r) for r in rows)
                new_collisions = get_collision_count(diff_counts)
                
                old_cost = current_collisions * 10000 + current_scope
                new_cost = new_collisions * 10000 + new_scope
                
                if new_cost < old_cost or rng.random() < math.exp(-(new_cost - old_cost) / max(T, min_T)):
                    current_scope = new_scope
                    current_collisions = new_collisions
                    if new_collisions == 0 and new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                        best_counts = dict(diff_counts)
                        if _BEST is None or best_scope < _BEST[0]:
                            _BEST = (best_scope, best_rows)
                else:
                    # Revert
                    _remove_row_diffs(diff_counts, rows[i])
                    rows[i] = old_row
                    _add_row_diffs(diff_counts, old_row)
            else:
                # Revert
                rows[i] = old_row
                _add_row_diffs(diff_counts, old_row)
                
        elif r < 0.75:
            # Swap two rows
            if n_rows < 2:
                continue
            i, j = rng.sample(range(n_rows), 2)
            rows[i], rows[j] = rows[j], rows[i]
            # Swapping rows doesn't change diff counts
            new_scope = max(max(r) for r in rows)
            new_collisions = current_collisions
            old_cost = current_collisions * 10000 + current_scope
            new_cost = new_collisions * 10000 + new_scope
            
            if new_cost < old_cost or rng.random() < math.exp(-(new_cost - old_cost) / max(T, min_T)):
                current_scope = new_scope
            else:
                rows[i], rows[j] = rows[j], rows[i]
                
        else:
            # Multi-element row shift
            i = rng.randint(0, n_rows - 1)
            start = rng.randint(1, k)
            end = rng.randint(start, k)
            shift = rng.randint(-3, 3)
            if shift == 0:
                continue
            
            old_block = list(rows[i][start:end + 1])
            new_block = [v + shift for v in old_block]
            
            # Check constraints
            valid = True
            if start > 0 and new_block[0] <= rows[i][start - 1]:
                valid = False
            if end < k and new_block[-1] >= rows[i][end + 1]:
                valid = False
            for bi in range(1, len(new_block)):
                if new_block[bi] <= new_block[bi - 1]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Remove old row's diffs
            _remove_row_diffs(diff_counts, rows[i])
            
            # Apply shift
            for bi, pos in enumerate(range(start, end + 1)):
                rows[i][pos] = new_block[bi]
            
            # Add new row's diffs
            _add_row_diffs(diff_counts, rows[i])
            
            new_scope = max(max(r) for r in rows)
            new_collisions = get_collision_count(diff_counts)
            
            old_cost = current_collisions * 10000 + current_scope
            new_cost = new_collisions * 10000 + new_scope
            
            if new_cost < old_cost or rng.random() < math.exp(-(new_cost - old_cost) / max(T, min_T)):
                current_scope = new_scope
                current_collisions = new_collisions
                if new_collisions == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                    best_counts = dict(diff_counts)
                    if _BEST is None or best_scope < _BEST[0]:
                        _BEST = (best_scope, best_rows)
            else:
                # Revert
                _remove_row_diffs(diff_counts, rows[i])
                for bi, pos in enumerate(range(start, end + 1)):
                    rows[i][pos] = old_block[bi]
                _add_row_diffs(diff_counts, rows[i])
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _polish_pass(rows, k, deadline, rng):
    """Final high-precision polish: try small perturbations on valid solution."""
    global _BEST
    
    if not _valid(rows, k):
        return rows
    
    current_scope = _scope(rows)
    
    for i in range(len(rows)):
        for j in range(1, k + 1):
            old_val = rows[i][j]
            prev_val = rows[i][j - 1]
            next_val = rows[i][j + 1] if j < k else old_val + 100
            
            for delta in range(-2, 3):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= prev_val or new_val >= next_val:
                    continue
                
                rows[i][j] = new_val
                if _valid(rows, k):
                    new_scope = _scope(rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if _BEST is None or new_scope < _BEST[0]:
                            _BEST = (new_scope, [list(r) for r in rows])
                else:
                    rows[i][j] = old_val
                if time.time() > deadline:
                    return rows
    
    return rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 145.0  # 5s safety margin
    
    rng = random.Random(42)
    
    # Generate multiple initial candidates
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
                max_step = rng.randint(1, 10)
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
    
    # Set initial best from best greedy
    best_scope = _scope(candidates[0])
    _BEST = (best_scope, [list(r) for r in candidates[0]])
    
    # Run SA on top candidates
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE - 10:
            break
        
        remaining = _DEADLINE - 10 - time.time()
        if remaining < 15:
            break
        
        sa_deadline = time.time() + remaining * (0.5 if idx == 0 else 0.3 if idx == 1 else 0.2)
        s, rows = _sa_optimize([list(r) for r in cand], k, n_rows, sa_deadline, rng)
        
        if _valid(rows, k):
            sc = _scope(rows)
            if _BEST is None or sc < _BEST[0]:
                _BEST = (sc, [list(r) for r in rows])
    
    # Polish pass
    if _BEST is not None and time.time() < _DEADLINE - 3:
        polish_deadline = _DEADLINE - 3
        rows = _polish_pass([list(r) for r in _BEST[1]], k, polish_deadline, rng)
        if _valid(rows, k):
            sc = _scope(rows)
            if _BEST is None or sc < _BEST[0]:
                _BEST = (sc, [list(r) for r in rows])
    
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
