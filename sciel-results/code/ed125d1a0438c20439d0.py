import time
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


# EVOLVE-BLOCK-START

def _all_diffs(rows):
    """Return the set of all positive differences across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(row[j] - row[i])
    return used


def _is_valid(rows):
    """Check if the set of rows is valid (all diffs distinct globally)."""
    used = set()
    for row in rows:
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        if diffs & used:
            return False
        used |= diffs
    return True


def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _gen_valid_rows(k, max_val, limit=2000000, time_budget=30.0):
    """
    Generate valid rows (k+1 increasing integers starting at 0, all <= max_val)
    with no internal difference collisions.
    Returns list of (row_tuple, frozenset_diffs, max_entry).
    """
    start = time.time()
    valid_rows = []
    target = k + 1

    def backtrack(pos, current, diffs):
        if len(valid_rows) >= limit:
            return
        if time.time() - start > time_budget:
            return
        if pos == target:
            valid_rows.append((tuple(current), frozenset(diffs), current[-1]))
            return
        if pos == 0:
            current = [0]
            diffs = set()
            backtrack(1, current, diffs)
            return

        prev = current[-1]
        remaining = target - pos
        min_next = prev + 1
        max_next = max_val - (remaining - 1)

        for x in range(min_next, max_next + 1):
            ok = True
            new_diffs = set()
            for a in current:
                d = x - a
                if d in diffs or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                current.append(x)
                diffs |= new_diffs
                backtrack(pos + 1, current, diffs)
                diffs -= new_diffs
                current.pop()

    backtrack(0, [], set())
    return valid_rows


def _greedy_init(n_rows, k):
    """Greedy initialization: each row takes the smallest next mark with no difference collision."""
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if len(new_diffs) != len(row):
                continue
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows


def _assignment_solve(rows_pool, n_rows):
    """
    Solve the minimum-cost assignment: assign n_rows rows from the pool such that
    no two assigned rows share a difference, minimizing the sum of max entries.
    Returns (assigned_rows, total_cost) or (None, None).
    """
    n = len(rows_pool)
    if n < n_rows:
        return None, None
    if n == n_rows:
        # Only one choice per row; just check conflict
        for i in range(n_rows):
            for j in range(i + 1, n_rows):
                if rows_pool[i][1] & rows_pool[j][1]:
                    return None, None
        return [list(rows_pool[i][0]) for i in range(n_rows)], sum(rows_pool[i][2] for i in range(n_rows))

    # Build conflict matrix
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rows_pool[i][1] & rows_pool[j][1]:
                conflict[i][j] = True
                conflict[j][i] = True

    BIG = 10 ** 9
    # Cost matrix: cost[i][j] = max_entry of pool[j] if compatible, else BIG
    cost = [[0.0] * n for _ in range(n)]
    for i in range(n_rows):
        for j in range(n):
            cost[i][j] = float(rows_pool[j][2])

    # We need to pick n_rows distinct pool indices for the n_rows rows,
    # with no conflicts between the picked ones, minimizing sum of max entries.
    # This is a minimum-weight independent set of size n_rows in the conflict graph.
    # Formulate as assignment: rows i in {0..n_rows-1} -> pool indices.
    # But conflicts are between pool indices, not between (row, pool) pairs.
    # So we need: assign each row a distinct pool index, and the set of chosen pool
    # indices must be independent in the conflict graph.

    # This is NOT a standard assignment (conflict depends on the pair of chosen pools).
    # Instead, use a different formulation: choose an independent set of size n_rows
    # minimizing sum of costs. We solve via DP/backtracking with pruning, or via
    # integer programming. For small n_rows, backtrack with branch-and-bound.

    # Branch and bound: pick n_rows pool indices forming an independent set, min sum cost.
    best = [None] * n_rows
    best_cost = [BIG]

    # Sort pool by max entry (ascending) for better pruning
    order = sorted(range(n), key=lambda j: rows_pool[j][2])

    def bb(idx, chosen, used, cost_so_far, max_idx_in_chosen):
        if cost_so_far >= best_cost[0]:
            return
        if idx == n_rows:
            best_cost[0] = cost_so_far
            best[:] = chosen[:]
            return
        # Pruning: even taking the (n_rows - idx) cheapest remaining compatible pools
        # can't beat best
        # (loose bound; skip for speed)

        # Try adding pool indices compatible with all chosen
        # Start from max_idx_inchosen+1 to avoid duplicates (since order is fixed)
        for j in range(max_idx_inchosen + 1, n):
            pool_j = order[j]
            if pool_j in used:
                continue
            # Check compatibility with all chosen
            ok = True
            for c in chosen:
                if conflict[pool_j][c]:
                    ok = False
                    break
            if not ok:
                continue
            chosen.append(pool_j)
            used.add(pool_j)
            bb(idx + 1, chosen, used, cost_so_far + rows_pool[pool_j][2], j)
            chosen.pop()
            used.remove(pool_j)

    bb(0, [], set(), 0, -1)

    if best_cost[0] >= BIG:
        return None, None

    assigned = [list(rows_pool[best[i]][0]) for i in range(n_rows)]
    return assigned, best_cost[0]


def _match_based_search(n_rows, k, time_budget):
    """
    Main search:
    1. Greedy init to get an initial scope M0.
    2. Generate a pool of valid rows up to M0 (and a bit beyond).
    3. Solve the assignment to minimize sum of max entries.
    4. Iterate: shrink the pool to the current best global max, regenerate, re-solve.
    """
    start_time = time.time()
    deadline = start_time + time_budget

    # 1. Greedy init
    greedy_rows = _greedy_init(n_rows, k)
    best_rows = [list(r) for r in greedy_rows]
    best_val = max(max(r) for r in greedy_rows)
    best_sum = sum(max(r) for r in best_rows)

    # 2. Generate initial pool
    pool_max = best_val
    pool = _gen_valid_rows(k, pool_max, limit=500000, time_budget=20.0)

    # 3. Solve assignment
    while time.time() < deadline - 10.0:
        assigned, total_cost = _assignment_solve(pool, n_rows)
        if assigned is not None:
            new_val = max(max(r) for r in assigned)
            if new_val < best_val or (new_val == best_val and total_cost < best_sum):
                best_val = new_val
                best_sum = total_cost
                best_rows = [list(r) for r in assigned]

        # 4. Shrink pool to current best and regenerate (to get more small rows)
        new_pool_max = best_val
        # Try to generate a denser pool up to best_val
        if time.time() < deadline - 15.0:
            pool = _gen_valid_rows(k, new_pool_max, limit=1000000, time_budget=25.0)
        else:
            break

    return best_rows, best_val


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Greedy initialization
    greedy_rows = _greedy_init(n_rows, k)
    best_rows = [list(r) for r in greedy_rows]
    best_val = max(max(r) for r in greedy_rows)

    # 2. Match-based search
    remaining = deadline - time.time() - 10.0
    if remaining > 20.0:
        result_rows, result_val = _match_based_search(n_rows, k, remaining)
        if result_val < best_val:
            best_val = result_val
            best_rows = [list(r) for r in result_rows]

    # Final validity check
    if not _is_valid(best_rows):
        if _is_valid(greedy_rows):
            return [list(r) for r in greedy_rows]
        # Fallback: construct a trivial valid set
        rows = []
        used = set()
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if len(new_diffs) != len(row):
                    continue
                if new_diffs & used:
                    continue
                row.append(x)
                used |= new_diffs
            rows.append(row)
        return rows

    return best_rows

# EVOLVE-BLOCK-END
