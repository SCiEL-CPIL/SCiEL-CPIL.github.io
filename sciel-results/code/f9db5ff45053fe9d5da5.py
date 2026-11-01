"""Difference triangle set (4,7) — constructive search."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

_DEADLINE = 118.0  # safety margin before 120s wall-clock

def _validate(rows):
    """Check that all positive differences across all rows are distinct."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _score(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_row(used, k):
    """Build a single row greedily: smallest next mark whose diffs are unused."""
    row = [0]
    x = 1
    while len(row) < k + 1:
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
        x += 1
    return row

def _greedy_construct(n_rows, k):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows

def _local_search(rows, k, max_iter=500, seed=0):
    """Local search: try to swap or adjust entries to reduce max entry.
    Returns (rows, improved) where improved is True if we found a better solution."""
    rng = random.Random(seed)
    best = [r[:] for r in rows]
    best_score = _score(best)
    
    for it in range(max_iter):
        # Pick a random row
        ri = rng.randrange(len(rows))
        # Pick a random position in that row (not 0, not the last)
        pos = rng.randrange(1, k)
        old_val = rows[ri][pos]
        # Try replacing with a nearby value
        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        if new_val <= rows[ri][pos - 1] or (pos + 1 < len(rows[ri]) and new_val >= rows[ri][pos + 1]):
            continue
        # Temporarily replace
        rows[ri][pos] = new_val
        # Check validity
        if _validate(rows):
            sc = _score(rows)
            if sc < best_score:
                best = [r[:] for r in rows]
                best_score = sc
            else:
                # Revert (accept only improvements for now; could add simulated annealing)
                rows[ri][pos] = old_val
        else:
            rows[ri][pos] = old_val
        if it % 100 == 0 and time.time() > _DEADLINE:
            break
    
    return best, best_score

def _multi_start_greedy(n_rows, k, num_starts=20, seed=0):
    """Try multiple random perturbations of greedy to find better solutions."""
    rng = random.Random(seed)
    best_rows = None
    best_score = float('inf')
    
    for start in range(num_starts):
        if time.time() > _DEADLINE:
            break
        # Start with greedy, then random local search
        used = set()
        rows = []
        # Build first row greedily
        row = _greedy_row(used, k)
        rows.append(row)
        # Build remaining rows with some randomness
        for _ in range(1, n_rows):
            row, x = [0], 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not new & used:
                    # With some probability, skip this candidate
                    if rng.random() < 0.3:
                        x += 1
                        continue
                    row.append(x)
                    used |= new
                x += 1
            rows.append(row)
        
        sc = _score(rows)
        if sc < best_score:
            best_score = sc
            best_rows = [r[:] for r in rows]
        
        # Local search to improve
        improved_rows, improved_score = _local_search(rows, k, max_iter=200, seed=start)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows
    
    return best_rows, best_score

def _construct_special_4_7():
    """
    Specialized construction for n_rows=4, k=7.
    Use a more systematic search: build rows with backtracking and pruning.
    """
    k = 7
    n_rows = 4
    
    # The counting bound says m >= 112. Best known is 134.
    # Let's try to find something better than greedy (which gives ~140+ for 4x8).
    
    # Strategy: Use a smarter ordering. Instead of always greedy,
    # try to balance the rows so they share the "load" of differences.
    
    # One approach: build all 4 rows simultaneously, always extending
    # the row that is "furthest behind" (smallest last element) or
    # the row that has the most "pressure" (most diffs used).
    
    def _balanced_construct():
        used = set()
        rows = [[0]] * 4
        for _ in range(k):
            # For each row, find the smallest candidate that works
            candidates = []
            for ri in range(4):
                last = rows[ri][-1]
                x = last + 1
                while True:
                    new = {x - a for a in rows[ri]}
                    if not new & used:
                        candidates.append((x, ri))
                        break
                    x += 1
            # Sort by candidate value (smallest first) — greedy balanced
            candidates.sort()
            for x, ri in candidates:
                rows[ri].append(x)
                used |= {x - a for a in rows[ri]}
        return rows
    
    def _backtrack_construct(max_val=140):
        """Backtracking: assign values to positions, prune when conflict."""
        # positions[row][col] = value
        pos = [[0] * (k + 1) for _ in range(n_rows)]
        used = set()
        
        def _dfs(row_idx, col_idx):
            if time.time() > _DEADLINE:
                return False
            if row_idx >= n_rows:
                return True
            if col_idx == k + 1:
                return _dfs(row_idx + 1, 0)
            
            if col_idx == 0:
                pos[row_idx][0] = 0
                return _dfs(row_idx, 1)
            
            # Determine range to try
            start = pos[row_idx][col_idx - 1] + 1
            end = max_val
            # Pruning: the remaining (k+1-col_idx) positions need increasing values
            # and the max must be <= max_val
            # Also, the last position must be <= max_val
            
            for v in range(start, end + 1):
                # Check: v - pos[row_idx][i] for i < col_idx must not be in used
                ok = True
                new_diffs = set()
                for i in range(col_idx):
                    d = v - pos[row_idx][i]
                    if d in used:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                # Check that remaining positions can fit
                # Need (k+1 - col_idx - 1) more values, each >= v+1, max <= max_val
                remaining = k + 1 - col_idx - 1
                if v + remaining > max_val:
                    break  # no point trying larger v either
                pos[row_idx][col_idx] = v
                old_used = set(used)
                used |= new_diffs
                if _dfs(row_idx, col_idx + 1):
                    return True
                used = old_used
            return False
        
        if _dfs(0, 0):
            return [list(r) for r in pos]
        return None
    
    # First try balanced greedy
    best_rows = _balanced_construct()
    best_score = _score(best_rows)
    
    # Try backtracking with decreasing max_val
    # Start from best_score and try to go lower
    for mv in range(best_score - 1, 111, -1):
        if time.time() > _DEADLINE - 10:
            break
        result = _backtrack_construct(mv)
        if result is not None:
            best_rows = result
            best_score = mv
            break
    
    # Local search on the best found
    improved_rows, improved_score = _local_search(best_rows, k, max_iter=300, seed=42)
    if improved_score < best_score:
        best_rows = improved_rows
        best_score = improved_score
    
    return best_rows

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    with all positive differences distinct across all rows.
    Minimize the largest entry."""
    t0 = time.time()
    
    # For the specific case (4,7), use specialized construction
    if n_rows == 4 and k == 7:
        rows = _construct_special_4_7()
        if rows is not None and _validate(rows):
            return rows
    
    # General case: multi-start greedy + local search
    # First, pure greedy as a baseline
    greedy_rows = _greedy_construct(n_rows, k)
    if not _validate(greedy_rows):
        # Shouldn't happen, but safety
        return greedy_rows
    
    best_rows = greedy_rows
    best_score = _score(best_rows)
    
    # Multi-start with randomness
    if time.time() < _DEADLINE - 20:
        ms_rows, ms_score = _multi_start_greedy(n_rows, k, num_starts=15, seed=123)
        if ms_rows is not None:
            if _validate(ms_rows) and ms_score < best_score:
                best_rows = ms_rows
                best_score = ms_score
    
    # Final local search polish
    if time.time() < _DEADLINE - 10:
        ls_rows, ls_score = _local_search(best_rows, k, max_iter=500, seed=999)
        if _validate(ls_rows) and ls_score < best_score:
            best_rows = ls_rows
    
    return best_rows
# EVOLVE-BLOCK-END
