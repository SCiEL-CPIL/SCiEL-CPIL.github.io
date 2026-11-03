"""Difference triangle set solver: Efficient SA with diff-counting + row reconstruction."""
import time
import sys
import random
import math

sys.setrecursionlimit(100000)


def _validate(rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _greedy(n_rows, k):
    """Fast greedy: smallest next mark with no difference collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            ok = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k, rng, bias=1.0):
    """Greedy with randomization: sometimes skip the smallest available value."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # With some probability, try a slightly larger value
            if bias > 0 and rng.random() < 0.15:
                x += rng.randint(1, int(bias) + 2)
            ok = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows


def _get_diffs(row):
    """Return list of all pairwise differences in a row."""
    diffs = []
    L = len(row)
    for i in range(L):
        for j in range(i + 1, L):
            diffs.append(row[j] - row[i])
    return diffs


def _reconstruct_row(r_idx, rows, used_diffs, upper_bound, row_len, rng=None):
    """Re-greedy row r_idx against the current set of used differences from other rows.
    If rng is provided, uses randomized greedy (skipping some valid values).
    """
    current_row = rows[r_idx]
    current_diffs = set(_get_diffs(current_row))
    for d in current_diffs:
        used_diffs.discard(d)

    new_row = [0]
    x = 0
    while len(new_row) < row_len:
        x += 1
        if x > upper_bound:
            break
        new_diffs = {x - a for a in new_row}
        if not (new_diffs & used_diffs):
            # If rng is provided, randomly skip valid values to diversify
            if rng is not None and rng.random() < 0.05:
                continue
            new_row.append(x)
            used_diffs |= new_diffs

    if len(new_row) == row_len:
        rows[r_idx] = new_row
        return True
    else:
        # Rollback
        for d in set(_get_diffs(new_row)):
            used_diffs.discard(d)
        used_diffs |= current_diffs
        return False


def _sa_row_swap(n_rows, k, initial_rows, deadline, seed=42):
    """Simulated Annealing with row-swap (reconstruction) moves."""
    rng = random.Random(seed)
    rows = [list(r) for r in initial_rows]
    L = k + 1  # entries per row

    # Build used_diffs set
    used_diffs = set()
    for r in rows:
        for d in _get_diffs(r):
            used_diffs.add(d)
            
    # We track the max value of each row to compute global scope efficiently
    row_maxs = [r[-1] for r in rows]
    current_scope = max(row_maxs)
    
    best_scope = current_scope
    best_rows = [r[:] for r in rows]

    T = 10.0
    T_min = 0.1
    cooling = 0.999

    start_time = time.time()
    
    while time.time() < deadline and T > T_min:
        # Pick a random row to reconstruct
        r_idx = rng.randrange(n_rows)
        old_row = rows[r_idx][:]
        old_max = row_maxs[r_idx]
        
        # Determine a bound for the new row. 
        # We want to try to shrink it, but allow expansion if it helps.
        bound = max(current_scope + 20, old_max + 10)
        
        # Reconstruct
        if _reconstruct_row(r_idx, rows, used_diffs, bound, L, rng):
            new_max = rows[r_idx][-1]
            new_diffs_set = set(_get_diffs(rows[r_idx]))
            row_maxs[r_idx] = new_max
            
            # Calculate new global scope
            new_scope = max(row_maxs)
            
            delta = new_scope - current_scope
            
            if delta < 0 or (rng.random() < math.exp(-delta / T)):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Revert
                # 1. Remove new diffs from used_diffs
                for d in new_diffs_set:
                    used_diffs.discard(d)
                # 2. Restore old row
                rows[r_idx] = old_row
                # 3. Add old diffs back to used_diffs
                old_diffs_set = set(_get_diffs(old_row))
                used_diffs |= old_diffs_set
                row_maxs[r_idx] = old_max

        T *= cooling

    return best_rows


def _multi_start_search(n_rows, k, deadline, base_seed=0):
    """Multiple randomized greedy starts."""
    rng = random.Random(base_seed)
    best_rows = None
    best_scope = float('inf')
    
    start_time = time.time()
    num_starts = 0
    
    while time.time() < deadline - 5.0:
        num_starts += 1
        seed = base_seed + num_starts
        
        # Generate a randomized greedy solution
        bias = 1.0 + (num_starts % 5) * 0.5
        greedy_rows = _randomized_greedy(n_rows, k, random.Random(seed), bias=bias)
        scope = _scope(greedy_rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = greedy_rows
            
        # Early exit if we found a very good solution (heuristic)
        if best_scope <= 112:
            break

    if best_rows is None:
        best_rows = _greedy(n_rows, k)
        best_scope = _scope(best_rows)
    
    return best_rows, best_scope


def construct(n_rows, k):
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s
    row_len = k + 1

    # Phase 1: Multi-start greedy to get a good initial solution
    ms_deadline = min(deadline - 30.0, start + 10.0)
    best, best_scope = _multi_start_search(n_rows, k, ms_deadline, base_seed=42)

    # Phase 2: Row-Swap SA
    if time.time() < deadline - 5.0:
        try:
            sa_deadline = deadline - 5.0
            sa_rows = _sa_row_swap(n_rows, k, best, sa_deadline, seed=42)
            if sa_rows is not None and _validate(sa_rows, k):
                sa_scope = _scope(sa_rows)
                if sa_scope < best_scope:
                    best = sa_rows
                    best_scope = sa_scope
        except Exception:
            pass

    # Phase 3: Final polish with a different seed and shorter time if available
    if time.time() < deadline - 2.0:
        try:
            sa2_deadline = deadline - 2.0
            sa2_rows = _sa_row_swap(n_rows, k, best, sa2_deadline, seed=999)
            if sa2_rows is not None and _validate(sa2_rows, k):
                sa2_scope = _scope(sa2_rows)
                if sa2_scope < best_scope:
                    best = sa2_rows
                    best_scope = sa2_scope
        except Exception:
            pass

    # Final validation
    if not _validate(best, k):
        best = _greedy(n_rows, k)

    return best
