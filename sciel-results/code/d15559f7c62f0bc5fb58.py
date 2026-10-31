"""Difference triangle set (4,7): minimize the largest entry (scope)."""
import time
import random
import sys

# We have a hard wall-clock budget; the evaluator kills at 150s, so keep a margin.
_DEADLINE = 145.0


def _all_diffs(row):
    """Return the set of all positive differences a_j - a_i (i<j) in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s


def _conflict_count(diff_set, used):
    """Count how many differences in diff_set are already in used."""
    c = 0
    for d in diff_set:
        if d in used:
            c += 1
    return c


def _greedy_row(used, k, max_val, rng):
    """
    Build one row of k+1 increasing integers starting at 0, with all entries <= max_val,
    such that its differences do not collide with 'used'.
    Uses a randomized greedy: at each step pick the smallest feasible next value,
    but with a small probability pick a slightly larger feasible value to diversify.
    Returns the row (list) or None if no feasible row exists within max_val.
    """
    row = [0]
    used_local = set(used)  # we track conflicts against global used only; within-row diffs are always distinct
    # Actually within-row diffs are distinct by construction (increasing sequence => unique pairwise diffs? No!)
    # Wait: in a single row, differences a_j - a_i for i<j are NOT necessarily all distinct.
    # Example: [0,1,2] gives diffs 1,2,1 -> duplicate 1.
    # The task says: "all differences between two entries of the same row are distinct over all rows"
    # Re-reading: "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
    # This means: the multiset of ALL differences across ALL rows must have no duplicates.
    # So within a single row, differences must also be distinct.
    # For a strictly increasing sequence, a_j - a_i = a_l - a_m can happen.
    # So we need each row to be a "Golomb ruler" (all pairwise differences distinct), AND
    # the difference sets of all rows must be pairwise disjoint.

    # So we need: each row is a Golomb ruler, and the diff sets are disjoint across rows.

    # Let's redo: build a Golomb ruler row whose diffs don't intersect 'used'.
    row = [0]
    row_diffs = set()
    for _ in range(k):
        # Find feasible next values
        best_val = None
        best_diffs = None
        best_conflict = None
        # We want the smallest x > row[-1] such that:
        #   1. {x - a for a in row} are all distinct (Golomb condition within row)
        #   2. {x - a for a in row} are disjoint from row_diffs (Golomb condition within row)
        #   3. {x - a for a in row} are disjoint from 'used' (global condition)
        #   4. x <= max_val
        # For diversity, sometimes pick a non-minimal feasible x.
        candidates = []
        x = row[-1] + 1
        while x <= max_val:
            new_diffs = set()
            ok = True
            for a in row:
                d = x - a
                if d in new_diffs or d in row_diffs or d in used:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                candidates.append((x, new_diffs))
            x += 1
        if not candidates:
            return None
        # Pick: mostly the smallest, sometimes a random one from the first few
        if rng.random() < 0.15 and len(candidates) > 3:
            # Pick randomly from the first 10 candidates for diversity
            idx = rng.randint(0, min(9, len(candidates) - 1))
            best_val, best_diffs = candidates[idx]
        else:
            best_val, best_diffs = candidates[0]
        row.append(best_val)
        row_diffs |= best_diffs
    return row


def _greedy_construct(n_rows, k, rng, max_val_hint=None):
    """
    Build n_rows Golomb rulers with disjoint difference sets, minimizing the max entry.
    Greedy: build rows one by one, each as small as possible.
    """
    used = set()
    rows = []
    for i in range(n_rows):
        # Start with a reasonable max_val and increase if needed
        if max_val_hint is not None:
            mv = max_val_hint
        else:
            # Estimate: for n_rows=4, k=7, best known is 134. Use a heuristic.
            mv = 100 + i * 20
        row = _greedy_row(used, k, mv, rng)
        if row is None:
            # Try larger max_val
            mv = 200
            while mv < 500:
                row = _greedy_row(used, k, mv, rng)
                if row is not None:
                    break
                mv += 50
        if row is None:
            # Fallback: very large
            row = _greedy_row(used, k, 1000, rng)
            if row is None:
                return None
        rows.append(row)
        used |= _all_diffs(row)
    return rows


def _scope(rows):
    return max(max(r) for r in rows)


def _is_valid(rows, k):
    """Check that all rows are valid Golomb rulers with disjoint diff sets."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if row[j] <= row[i]:
                    return False
        diffs = _all_diffs(row)
        if len(diffs) != len(row) * (len(row) - 1) // 2:
            return False  # duplicate within row
        if diffs & used:
            return False  # collision across rows
        used |= diffs
    return True


def _local_search(rows, k, rng, time_limit):
    """
    Local search: try to reduce the scope by moving individual entries.
    """
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    start = time.time()
    
    # Precompute diff sets
    def get_all_diffs(rows):
        used = set()
        for row in rows:
            used |= _all_diffs(row)
        return used
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try to reduce the largest entry
        # Find the row with the max entry
        max_row_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        max_val = rows[max_row_idx][-1]
        
        # Try to reduce entries in the max row
        for pos in range(len(rows[max_row_idx]) - 1, 0, -1):
            old_val = rows[max_row_idx][pos]
            # Try smaller values
            for new_val in range(rows[max_row_idx][pos - 1] + 1, old_val):
                rows[max_row_idx][pos] = new_val
                # Check validity
                if _is_valid(rows, k):
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        break
                rows[max_row_idx][pos] = old_val
            if improved:
                break
        if not improved:
            # Try random perturbations
            if (time.time() - start) < time_limit:
                for _ in range(50):
                    if (time.time() - start) >= time_limit:
                        break
                    ri = rng.randint(0, len(rows) - 1)
                    pi = rng.randint(1, len(rows[ri]) - 1)
                    old_val = rows[ri][pi]
                    delta = rng.randint(-5, 5)
                    new_val = old_val + delta
                    if new_val <= rows[ri][pi - 1] or new_val >= (rows[ri][pi + 1] if pi + 1 < len(rows[ri]) else float('inf')):
                        continue
                    rows[ri][pi] = new_val
                    if _is_valid(rows, k):
                        new_scope = _scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            improved = True
                        # Accept with some probability even if not better (simulated annealing-like)
                        elif rng.random() < 0.1:
                            pass  # keep it
                    else:
                        rows[ri][pi] = old_val
    return rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    with all positive differences distinct over all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    start_time = time.time()
    rng = random.Random(42)
    
    # Get a baseline solution
    baseline = _greedy_construct(n_rows, k, rng)
    if baseline is None:
        # Fallback: very simple greedy without max_val constraint
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not new & used and len(new) == len(row):
                        break
                    x += 1
                row.append(x)
                used |= new
            rows.append(row)
        baseline = rows
    
    best_rows = baseline
    best_scope = _scope(best_rows)
    
    # Try multiple random restarts with local search
    time_budget = _DEADLINE - start_time
    restart_time = time_budget / 5.0  # time per restart including local search
    
    for restart in range(10):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        # Randomized greedy construction
        rows = _greedy_construct(n_rows, k, rng)
        if rows is None:
            continue
        
        scope = _scope(rows)
        
        # Local search to improve
        local_time = min(restart_time - (time.time() - start_time), 20.0)
        if local_time > 2:
            rows, scope = _local_search(rows, k, rng, local_time)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    # Final local search on the best solution with remaining time
    remaining = _DEADLINE - 5 - (time.time() - start_time)
    if remaining > 2:
        best_rows, best_scope = _local_search(best_rows, k, rng, remaining)
    
    # Verify validity
    if not _is_valid(best_rows, k):
        # Fallback to a known-valid solution
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not new & used and len(new) == len(row):
                        break
                    x += 1
                row.append(x)
                used |= new
            rows.append(row)
        best_rows = rows
    
    return best_rows
