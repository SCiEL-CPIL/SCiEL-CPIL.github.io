"""Difference triangle set (n_rows, k) with deadline-guarded search."""
import time
import random

_DEADLINE_MARGIN = 5.0  # seconds to stay under the 150s kill


def _row_diffs(row):
    """Return the set of positive differences within a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _scope(rows):
    """Largest entry across all rows."""
    return max(max(r) for r in rows) if rows else 0


def _is_valid(rows, k):
    """Check that all rows have k+1 entries starting at 0, strictly increasing,
    and all positive differences are distinct over all rows."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = _row_diffs(row)
        if diffs & used:
            return False
        used |= diffs
    return True


def _greedy_row(used, k, start_hint=0):
    """Build one row greedily: pick smallest x > last such that no diff collision."""
    row = [0]
    x = max(start_hint, 1)
    while len(row) < k + 1:
        while True:
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                break
            x += 1
        row.append(x)
        used |= {x - a for a in row[:-1]}
        x += 1
    return row


def _greedy_full(n_rows, k):
    """Original greedy: build rows in order, each as small as possible."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows


def _greedy_with_shuffles(n_rows, k, rng, max_iters=200):
    """Try building rows in different orders and with random perturbations."""
    best_rows = None
    best_scope = float('inf')
    for _ in range(max_iters):
        used = set()
        rows = []
        order = list(range(n_rows))
        rng.shuffle(order)
        offsets = [0] * n_rows
        for _r in range(n_rows):
            if rng.random() < 0.3:
                offsets[_r] = rng.randint(0, 3)
        row_by_idx = {}
        for idx in order:
            row = _greedy_row(used, k, start_hint=offsets[idx])
            row_by_idx[idx] = row
        rows = [row_by_idx[i] for i in range(n_rows)]
        s = _scope(rows)
        if s < best_scope:
            best_scope = s
            best_rows = rows
    return best_rows


def _compact_row(row, used, k):
    """Try to compact a single row by reducing entries from the end."""
    row = list(row)
    # Recompute diffs for this row
    row_diffs = _row_diffs(row)
    # Try to reduce each entry from the end
    for pos in range(len(row) - 1, 0, -1):
        old_val = row[pos]
        # Try smaller values
        for new_val in range(old_val - 1, row[pos - 1], -1):
            # Check if new_val is valid
            new_diffs = {new_val - a for a in row[:pos]}
            # Remove old diffs involving this position
            old_diffs = {old_val - a for a in row[:pos]}
            # Check collision with other row diffs
            other_diffs = used - row_diffs
            if not (new_diffs & other_diffs) and not (new_diffs & row_diffs - old_diffs):
                row[pos] = new_val
                row_diffs = row_diffs - old_diffs | new_diffs
                break
    return row


def _local_search(rows, k, rng, time_limit=10.0):
    """Attempt to reduce scope by compacting individual rows."""
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Collect all used diffs
        used = set()
        for r in rows:
            used |= _row_diffs(r)
        # Try to compact each row
        for ri in range(len(rows)):
            row = rows[ri]
            old_row = list(row)
            row = _compact_row(row, used, k)
            if row != old_row:
                # Update used diffs
                old_diffs = _row_diffs(old_row)
                new_diffs = _row_diffs(row)
                used = (used - old_diffs) | new_diffs
                rows[ri] = row
                improved = True
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
    return rows, _scope(rows)


def _simulated_annealing(rows, k, rng, time_limit=15.0):
    """Simulated annealing to reduce scope."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    start = time.time()
    current_temp = 10.0
    cooling = 0.995
    
    while time.time() - start < time_limit:
        # Compute current used diffs
        used = set()
        for r in rows:
            used |= _row_diffs(r)
        
        # Pick a random row and position
        ri = rng.randint(0, len(rows) - 1)
        row = rows[ri]
        pos = rng.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Try a random perturbation
        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        
        # Check validity
        if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
            continue
        
        # Check diff collisions
        new_diffs = {new_val - a for a in row[:pos]} | {a - new_val for a in row[pos + 1:]}
        old_diffs = {old_val - a for a in row[:pos]} | {a - old_val for a in row[pos + 1:]}
        remaining = used - old_diffs
        
        if new_diffs & remaining:
            continue
        
        # Compute new scope
        new_scope = max(max(r) for r in rows)
        new_scope = max(new_scope, new_val)
        old_scope = _scope(rows)
        
        delta_scope = new_scope - old_scope
        if delta_scope <= 0 or rng.random() < (1.0 / (1.0 + current_temp)):
            # Accept move
            used = remaining | new_diffs
            row[pos] = new_val
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        
        current_temp *= cooling
    
    return best_rows, best_scope


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct over all rows.
    Minimize the largest entry (scope)."""
    t0 = time.time()
    deadline = t0 + 150.0 - _DEADLINE_MARGIN

    rng = random.Random(42)

    # Phase 1: Basic greedy
    rows = _greedy_full(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)

    # Phase 2: Try shuffled orderings
    remaining = deadline - time.time()
    if remaining > 5.0:
        iters = min(200, max(10, int(remaining / 0.05)))
        shuffled = _greedy_with_shuffles(n_rows, k, rng, max_iters=iters)
        if shuffled is not None:
            s = _scope(shuffled)
            if s < best_scope:
                best_scope = s
                best_rows = shuffled

    # Phase 3: Local search on best solution
    remaining = deadline - time.time()
    if remaining > 2.0:
        ls_time = min(remaining - 1.0, 15.0)
        if ls_time > 1.0:
            improved_rows, s = _local_search(best_rows, k, rng, time_limit=ls_time)
            if s < best_scope:
                best_scope = s
                best_rows = improved_rows

    # Phase 4: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 5.0:
        sa_time = min(remaining - 1.0, 20.0)
        if sa_time > 2.0:
            sa_rows, s = _simulated_annealing(best_rows, k, rng, time_limit=sa_time)
            if s < best_scope:
                best_scope = s
                best_rows = sa_rows

    # Phase 5: More shuffled attempts if time remains
    remaining = deadline - time.time()
    if remaining > 5.0:
        rng2 = random.Random(123)
        iters = min(300, max(10, int(remaining / 0.05)))
        shuffled = _greedy_with_shuffles(n_rows, k, rng2, max_iters=iters)
        if shuffled is not None:
            s = _scope(shuffled)
            if s < best_scope:
                best_scope = s
                best_rows = shuffled
                # Another local search
                remaining = deadline - time.time()
                if remaining > 2.0:
                    ls_time = min(remaining - 1.0, 10.0)
                    if ls_time > 1.0:
                        ir, s2 = _local_search(best_rows, k, rng2, time_limit=ls_time)
                        if s2 < best_scope:
                            best_scope = s2
                            best_rows = ir

    return best_rows
