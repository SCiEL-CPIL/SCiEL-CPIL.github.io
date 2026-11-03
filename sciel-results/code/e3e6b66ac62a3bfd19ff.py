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

def _diffs(row):
    """Return list of all positive differences in a row."""
    return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]

def _greedy_build(n_rows, k, rng, tight=True):
    """
    Build rows one at a time using a greedy strategy.
    For each row, we try to build a valid row with minimal scope.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            best_x = None
            best_diffs = None
            best_score = float('inf')
            
            # Try values from x+1 upward
            # Limit search to a reasonable bound
            search_limit = x + 50 if tight else x + 200
            for candidate in range(x + 1, search_limit + 1):
                # Check if adding this candidate creates any conflict
                ok = True
                new_diffs = []
                for a in row:
                    d = candidate - a
                    if d in used:
                        ok = False
                        break
                    new_diffs.append(d)
                if ok:
                    # Score: prefer smaller values
                    score = candidate
                    # Also consider: how many future conflicts? Hard to predict
                    if score < best_score:
                        best_score = score
                        best_x = candidate
                        best_diffs = new_diffs
                        if tight:
                            break  # Take the first valid one
            if best_x is None:
                # Fallback: try wider search
                for candidate in range(x + 1, x + 1000):
                    ok = True
                    new_diffs = []
                    for a in row:
                        d = candidate - a
                        if d in used:
                            ok = False
                            break
                    new_diffs.append(d)
                    if ok:
                        best_x = candidate
                        best_diffs = new_diffs
                        break
            
            if best_x is None:
                # This shouldn't happen, but if it does, just pick a large value
                best_x = x + 1000
                best_diffs = [best_x - a for a in row]
            
            row.append(best_x)
            x = best_x
            for d in best_diffs:
                used.add(d)
        rows.append(row)
    return rows

def _backtrack_build(n_rows, k, scope_bound, rng, deadline):
    """
    Backtracking search for a set of rows with scope <= scope_bound.
    Returns (rows, success).
    """
    rows = []
    used = set()
    
    def backtrack(row_idx, row, pos, x):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            rows.append(list(row))
            if row_idx + 1 == n_rows:
                return True
            return backtrack(row_idx + 1, [0], 0, 0)
        
        # Try next value
        start = x + 1 if pos > 0 else 1
        for candidate in range(start, scope_bound + 1):
            # Check conflicts
            ok = True
            for a in row:
                d = candidate - a
                if d in used:
                    ok = False
                    break
            if ok:
                new_diffs = [candidate - a for a in row]
                for d in new_diffs:
                    used.add(d)
                row.append(candidate)
                
                if backtrack(row_idx, row, pos + 1, candidate):
                    return True
                
                row.pop()
                for d in new_diffs:
                    used.remove(d)
        
        return False
    
    if backtrack(0, [0], 0, 0):
        return rows, True
    return None, False

def _local_search(rows, k, deadline, rng):
    """
    Local search to reduce scope. Start from a valid solution and try to
    move elements to reduce the maximum value.
    """
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [list(r) for r in best_rows]
    
    # Build used set
    used = set()
    for r in current_rows:
        for d in _diffs(r):
            used.add(d)
    
    # Try to reduce scope by moving the largest element
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Find the element with the maximum value
        max_val = best_scope
        max_row = -1
        max_pos = -1
        for i, r in enumerate(current_rows):
            for j in range(len(r) - 1, 0, -1):
                if r[j] == max_val:
                    max_row = i
                    max_pos = j
                    break
            if max_row >= 0:
                break
        
        if max_row < 0:
            break
        
        row = current_rows[max_row]
        # Try to move this element to a smaller position
        # Remove its differences from used
        old_diffs = _diffs(row)
        for d in old_diffs:
            used.discard(d)
        
        old_val = row[max_pos]
        found = False
        # Try values from row[max_pos-1]+1 to old_val-1
        for new_val in range(row[max_pos - 1] + 1, old_val):
            # Check if new_val is valid
            ok = True
            new_diffs = []
            for j in range(len(row)):
                if j == max_pos:
                    continue
                d = abs(new_val - row[j])
                if d == 0:
                    ok = False
                    break
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                # Also check internal conflicts: new_diffs should be all distinct
                # and not equal to each other
                if len(set(new_diffs)) == len(new_diffs):
                    row[max_pos] = new_val
                    for d in new_diffs:
                        used.add(d)
                    found = True
                    break
        
        if not found:
            # Restore
            row[max_pos] = old_val
            for d in old_diffs:
                used.add(d)
            break
        
        # Update best
        new_scope = _scope(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [list(r) for r in current_rows]
        
        # Also try other random moves
        if rng.random() < 0.3:
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j - 1]
            next_val = current_rows[i][j + 1] if j < k else best_scope + 1
            
            # Remove row diffs
            row = current_rows[i]
            old_diffs = _diffs(row)
            for d in old_diffs:
                used.discard(d)
            
            # Try a few random values
            tried = False
            for _ in range(10):
                new_val = rng.randint(prev_val + 1, next_val - 1)
                if new_val == old_val:
                    continue
                ok = True
                new_diffs = []
                for t in range(k + 1):
                    if t == j:
                        continue
                    d = abs(new_val - current_rows[i][t])
                    if d == 0 or d in used:
                        ok = False
                        break
                    new_diffs.append(d)
                if ok and len(set(new_diffs)) == len(new_diffs):
                    current_rows[i][j] = new_val
                    for d in new_diffs:
                        used.add(d)
                    tried = True
                    break
            
            if not tried:
                # Restore
                current_rows[i][j] = old_val
                for d in old_diffs:
                    used.add(d)
            else:
                # Rebuild used set for this row is already done
                # Check if improved
                new_scope = _scope(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 140.0
    
    rng = random.Random(12345)
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
    
    # Generate initial candidates using greedy
    candidates = []
    for tight in [True, False]:
        for seed in range(3):
            r = random.Random(seed * 100 + tight * 10)
            rows = _greedy_build(n_rows, k, r, tight=tight)
            candidates.append(rows)
    
    # Sort by scope
    candidates.sort(key=_scope)
    
    # Try backtracking for tighter bounds
    for cand in candidates[:2]:
        if time.time() > _DEADLINE - 30:
            break
        current_scope = _scope(cand)
        # Try to find a solution with scope = current_scope - 1
        bound = current_scope - 1
        while bound >= 80 and time.time() < _DEADLINE - 10:
            bt_deadline = time.time() + min(5.0, _DEADLINE - time.time() - 5.0)
            rows, success = _backtrack_build(n_rows, k, bound, rng, bt_deadline)
            if success and rows:
                consider(rows)
                bound -= 1
            else:
                break
    
    # Local search to further reduce scope
    if _BEST is not None and time.time() < _DEADLINE - 10:
        best_rows = _BEST[1]
        best_scope = _BEST[0]
        
        # Run local search with time limit
        ls_deadline = time.time() + min(30.0, _DEADLINE - time.time() - 5.0)
        s, rows = _local_search(best_rows, k, ls_deadline, rng)
        consider(rows)
        
        # Run local search again with a different starting point
        if time.time() < _DEADLINE - 5:
            for cand in candidates[:3]:
                if time.time() > _DEADLINE - 5:
                    break
                ls_deadline2 = time.time() + min(10.0, _DEADLINE - time.time() - 2.0)
                s2, rows2 = _local_search(cand, k, ls_deadline2, rng)
                consider(rows2)
    
    if _BEST is not None:
        return _BEST[1]
    # Fallback: return the best candidate
    return candidates[0] if candidates else _greedy_build(n_rows, k, rng, tight=True)
