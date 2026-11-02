"""Difference triangle set (4,7) – constructive + local-search hybrid."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _all_diffs(row):
    """Return the set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _rows_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for d in _all_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _scope(rows):
    """The largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_row(used, k):
    """Build one row of k+1 entries starting at 0 using greedy smallest-available."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row


def _greedy_construct(n_rows, k):
    """Greedy construction: build rows one by one."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows


def _local_search(rows, k, max_iters=50000, deadline=None):
    """
    Local search: repeatedly pick a row and a position, try shifting a suffix
    or swapping two entries to reduce the global scope while maintaining validity.
    """
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    n_rows = len(rows)
    
    # Precompute: for a candidate set of rows, check validity and scope
    def try_move(candidate_rows):
        """Return (valid, scope) for candidate_rows."""
        seen = set()
        scope = 0
        for row in candidate_rows:
            for i in range(len(row)):
                if row[i] > scope:
                    scope = row[i]
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in seen:
                        return False, scope
                    seen.add(d)
        return True, scope
    
    rng = random.Random(42)
    iters = 0
    start = time.time()
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        iters += 1
        
        # Pick a random row
        ri = rng.randrange(n_rows)
        row = list(best_rows[ri])
        k1 = len(row)
        
        # Try a few mutation strategies
        improved = False
        
        # Strategy 1: shift a single entry (not the first)
        pos = rng.randrange(1, k1)
        orig = row[pos]
        # Try to move it to a smaller value
        for delta in range(1, min(orig, 30)):
            candidate = row[pos] - delta
            if candidate <= row[pos - 1]:
                break
            row[pos] = candidate
            # Check if this row is still strictly increasing
            ok = True
            for pp in range(1, k1):
                if row[pp] <= row[pp - 1]:
                    ok = False
                    break
            if not ok:
                row[pos] = orig
                continue
            # Check full validity
            candidate_rows = [list(r) for r in best_rows]
            candidate_rows[ri] = row
            valid, scope = try_move(candidate_rows)
            if valid and scope < best_scope:
                best_rows[ri] = row
                best_scope = scope
                improved = True
                break
            row[pos] = orig
        
        # Strategy 2: try to compact by shifting the last few entries down
        if not improved:
            # Try reducing the last entry
            pos = k1 - 1
            orig = row[pos]
            for delta in range(1, min(orig - row[pos-1], 20)):
                candidate = row[pos] - delta
                row[pos] = candidate
                valid, scope = try_move([best_rows[i] if i != ri else row for i in range(n_rows)])
                if valid and scope < best_scope:
                    best_rows[ri] = row
                    best_scope = scope
                    improved = True
                    break
                row[pos] = orig
        
        if not improved:
            # Try a more aggressive move: rebuild a random row with a different seed
            # by removing it and regreedy-ing with a slight perturbation
            if iters % 100 == 0:
                used = set()
                for i in range(n_rows):
                    if i == ri:
                        continue
                    used |= _all_diffs(best_rows[i])
                # Try to build a new row that fits in a smaller scope
                max_allowed = best_scope
                new_row = [0]
                x = 0
                success = False
                while len(new_row) < k1:
                    x += 1
                    if x > max_allowed:
                        break
                    new_diffs = {x - a for a in new_row}
                    if not new_diffs & used:
                        new_row.append(x)
                if len(new_row) == k1:
                    new_rows = [list(r) for r in best_rows]
                    new_rows[ri] = new_row
                    valid, scope = try_move(new_rows)
                    if valid and scope < best_scope:
                        best_rows[ri] = new_row
                        best_scope = scope
    
    return best_rows, best_scope


def _multi_start_greedy(n_rows, k, n_starts=10, deadline=None):
    """Try multiple greedy constructions with different row orderings."""
    best_rows = None
    best_scope = float('inf')
    
    for start in range(n_starts):
        if deadline is not None and time.time() > deadline:
            break
        used = set()
        rows = []
        # Shuffle which "slot" we fill first to get diversity
        order = list(range(n_rows))
        random.shuffle(order)
        row_slots = [None] * n_rows
        for slot in order:
            row = _greedy_row(used, k)
            row_slots[slot] = row
        rows = row_slots
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe under 150s limit
    
    # Phase 1: Greedy construction
    rows = _greedy_construct(n_rows, k)
    best_scope = _scope(rows)
    
    # Phase 2: Multi-start greedy
    ms_rows, ms_scope = _multi_start_greedy(n_rows, k, n_starts=20, deadline=deadline)
    if ms_rows is not None and ms_scope < best_scope:
        rows = ms_rows
        best_scope = ms_scope
    
    # Phase 3: Local search to compact
    if time.time() < deadline - 10:
        rows, ls_scope = _local_search(rows, k, max_iters=200000, deadline=deadline - 5)
        best_scope = ls_scope
    
    return rows

# EVOLVE-BLOCK-END
