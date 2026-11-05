"""
Difference triangle set (7,5) solver.

Strategy:
1. Generate a diverse pool of candidate rows (greedy, randomized greedy, random walk).
2. Build a conflict graph using bitmasks over the difference values (fast).
3. Greedy maximum-clique with multiple orderings + one branch-and-bound pass.
4. Fast local search / polish to minimize the scope (largest entry).
"""
import time
import random
import math

# EVOLVE-BLOCK-START

# Hard wall-clock limit for the evaluator; keep a safety margin.
_HARD_DEADLINE = 150.0
_MARGIN = 10.0


def _diffs_of_row(row):
    """Return a set of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs


def _total_scope(rows):
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used


def _generate_row_greedy_pure(used, k):
    """Pure greedy: always pick the smallest valid candidate."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, frozenset(new_diffs)


def _generate_row_greedy(used, k, rng, pool_size=3):
    """Greedily construct a row, picking among the top smallest valid candidates."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        if rng.random() < 0.5:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, frozenset(new_diffs)


def _generate_row_random_walk(k, rng, max_scope=150):
    """Random row of k+1 integers starting at 0, max entry <= max_scope."""
    row = [0]
    new_diffs = set()
    x = 0
    attempts = 0
    while len(row) < k + 1:
        attempts += 1
        if attempts > 2000:
            return None, None
        min_val = x + 1
        if min_val > max_scope:
            return None, None
        x = rng.randint(min_val, min(max_scope, x + rng.randint(1, 10)))
        new = {x - a for a in row}
        if new & new_diffs:
            continue
        row.append(x)
        new_diffs |= new
    return row, frozenset(new_diffs)


def _generate_candidate_pool(n_rows, k, rng, pool_size=200):
    """Generate a pool of candidate (row, diffs) tuples."""
    candidates = []

    # Base: pure greedy
    used = set()
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is not None:
            candidates.append((row, diffs))
            used |= diffs

    # Randomized greedy
    for _ in range(pool_size):
        used_subset = set()
        for c in candidates:
            if rng.random() < 0.3:
                used_subset |= c[1]
        ps = rng.choice([2, 3, 5, 8])
        row, diffs = _generate_row_greedy(used_subset, k, rng, pool_size=ps)
        if row is not None:
            candidates.append((row, diffs))

    # Random walk rows
    for _ in range(pool_size // 2):
        row, diffs = _generate_row_random_walk(k, rng, max_scope=120)
        if row is not None:
            candidates.append((row, diffs))

    return candidates


def _build_conflict_masks(candidates, max_diff):
    """
    For each candidate, build a bitmask over difference values [1, max_diff]
    indicating which differences it uses.
    """
    masks = []
    for _, diffs in candidates:
        m = 0
        for d in diffs:
            if 1 <= d <= max_diff:
                m |= (1 << d)
        masks.append(m)
    return masks


def _greedy_clique(candidates, masks, n_rows, order):
    """Greedy set packing: pick rows in `order`, keeping those with disjoint diffs."""
    chosen = []
    chosen_mask = 0
    for idx in order:
        if len(chosen) >= n_rows:
            break
        m = masks[idx]
        if (m & chosen_mask) == 0:
            chosen.append(idx)
            chosen_mask |= m
    return chosen


def _branch_and_bound_clique(candidates, masks, n_rows, deadline):
    """
    Branch and bound for a clique of size n_rows (rows with pairwise disjoint diffs).
    Returns a list of indices of size n_rows, or None.
    """
    n = len(candidates)
    if n < n_rows:
        return None
    # Order candidates by scope ascending (prefer small rows), tie-break by diff count.
    idx_order = sorted(range(n), key=lambda i: (candidates[i][0][-1], len(candidates[i][1])))
    # Reindex into compact list
    rem = list(idx_order)
    # Precompute pairwise compatibility using masks (disjoint = compatible)
    # We'll work in the compact index space.
    size = len(rem)
    comp = [0] * size  # bitmask of compatible candidates within compact space
    for i in range(size):
        mi = masks[rem[i]]
        bits = 0
        for j in range(size):
            if i == j:
                continue
            if (mi & masks[rem[j]]) == 0:
                bits |= (1 << j)
        comp[i] = bits

    best = None
    found = [False]

    def search(clique, avail):
        if found[0] or time.time() > deadline:
            return
        # avail is a bitmask (in compact space) of candidates that are compatible
        # with every member of the current clique.
        if len(clique) + avail.bit_count() < n_rows:
            return
        # Try to complete
        # Pick candidates from avail
        a = avail
        while a:
            if found[0] or time.time() > deadline:
                return
            # choose a candidate with fewest remaining options (branch on it)
            # pick lowest set bit for simplicity
            lb = a & (-a)
            i = lb.bit_length() - 1
            a ^= lb
            new_avail = avail & comp[i]
            clique.append(i)
            if len(clique) == n_rows:
                found[0] = True
                best[:] = [rem[c] for c in clique]
                return
            search(clique, new_avail)
            clique.pop()
            if found[0]:
                return

    search([], (1 << size) - 1)
    return best


def _local_search_scope(rows, k, rng, deadline):
    """Local search to reduce the scope (maximum entry)."""
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        if num_passes % 3 == 0:
            row_indices = list(range(len(rows)))
            rng.shuffle(row_indices)
        elif num_passes % 3 == 1:
            row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        else:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1])

        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]
            # Try to shrink each position from the largest down.
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                lo = row[pos - 1] + 1
                for new_val in range(old_val - 1, lo - 1, -1):
                    if new_val == old_val:
                        continue
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        improved = True
                        break
    return rows


def _polish_pass(rows, k, deadline):
    """High-precision polish: repeated full reductions."""
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

    for _ in range(6):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    lo = row[pos - 1] + 1
                    for new_val in range(old_val - 1, lo - 1, -1):
                        if new_val == old_val:
                            continue
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break
    return rows


def _simulated_annealing_scope(rows, k, rng, deadline):
    """Simulated annealing to escape local optima in scope minimization."""
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope

    temperature = 10.0
    cooling_rate = 0.995

    while time.time() < deadline and temperature > 0.1:
        idx = rng.randint(0, len(rows) - 1)
        pos = rng.randint(1, len(rows[idx]) - 1)
        row = rows[idx]
        old_val = row[pos]

        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        if new_val <= row[pos - 1]:
            continue
        if pos + 1 < len(row) and new_val >= row[pos + 1]:
            continue

        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)

        remaining = used - old_diffs
        if new_diffs & remaining:
            continue

        # new scope
        new_scope = max(current_scope, new_val)
        if new_val == current_scope:
            other_max = max(max(r) for i, r in enumerate(rows) if i != idx)
            new_scope = max(other_max, new_val)

        delta_scope = new_scope - current_scope
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]

        temperature *= cooling_rate

    return best_rows


def _multi_start_local_search(n_rows, k, rng, deadline):
    """Multiple restarts of local search with different initial solutions."""
    best_rows = None
    best_scope = float('inf')

    start_time = time.time()
    num_starts = 0

    while time.time() < deadline and num_starts < 8:
        num_starts += 1
        used = set()
        rows = []
        for _ in range(n_rows):
            if rng.random() < 0.5:
                row, diffs = _generate_row_greedy_pure(used, k)
            else:
                row, diffs = _generate_row_greedy(used, k, rng, pool_size=rng.choice([2, 3, 5]))
            if row is None:
                row, diffs = _generate_row_greedy_pure(used, k)
            if row is not None:
                rows.append(row)
                used |= diffs

        if len(rows) < n_rows:
            continue

        rows = _local_search_scope(rows, k, rng, deadline - 5)
        scope = _total_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    return best_rows


def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= diffs
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = min(_HARD_DEADLINE - _MARGIN, start_time + 140.0)

    # Step 1: Generate candidate pool
    candidates = _generate_candidate_pool(n_rows, k, rng, pool_size=300)

    # Deduplicate by diffs
    seen_diffs = set()
    unique_candidates = []
    for row, diffs in candidates:
        if diffs not in seen_diffs:
            seen_diffs.add(diffs)
            unique_candidates.append((row, diffs))
    candidates = unique_candidates

    # Compute max diff for bitmask sizing
    max_diff = 1
    for _, diffs in candidates:
        if diffs:
            max_diff = max(max_diff, max(diffs))

    rows = None

    if len(candidates) >= n_rows and time.time() < deadline - 40:
        masks = _build_conflict_masks(candidates, max_diff)

        # Try several greedy orderings
        best_clique = None
        best_clique_scope = float('inf')

        # Ordering 1: by scope ascending
        order1 = sorted(range(len(candidates)), key=lambda i: (candidates[i][0][-1], len(candidates[i][1])))
        c1 = _greedy_clique(candidates, masks, n_rows, order1)
        if len(c1) >= n_rows:
            scope = max(candidates[i][0][-1] for i in c1)
            if scope < best_clique_scope:
                best_clique_scope = scope
                best_clique = c1

        # Ordering 2: by diff-count ascending
        order2 = sorted(range(len(candidates)), key=lambda i: (len(candidates[i][1]), candidates[i][0][-1]))
        c2 = _greedy_clique(candidates, masks, n_rows, order2)
        if len(c2) >= n_rows:
            scope = max(candidates[i][0][-1] for i in c2)
            if scope < best_clique_scope:
                best_clique_scope = scope
                best_clique = c2

        # Ordering 3: random shuffle (a few times)
        for _ in range(20):
            order3 = list(range(len(candidates)))
            rng.shuffle(order3)
            c3 = _greedy_clique(candidates, masks, n_rows, order3)
            if len(c3) >= n_rows:
                scope = max(candidates[i][0][-1] for i in c3)
                if scope < best_clique_scope:
                    best_clique_scope = scope
                    best_clique = c3

        # Branch and bound if time permits
        if best_clique is None and time.time() < deadline - 30:
            bb = _branch_and_bound_clique(candidates, masks, n_rows, deadline - 30)
            if bb is not None and len(bb) >= n_rows:
                best_clique = bb

        if best_clique is not None:
            rows = [candidates[i][0] for i in best_clique]
        else:
            rows = _greedy_construct(n_rows, k)
            if rows is None:
                rows = []
                used = set()
                for _ in range(n_rows):
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        x += 1
                        new = {x - a for a in row}
                        if not (new & used):
                            row.append(x)
                            used |= new
                    rows.append(row)
                return rows
    else:
        rows = _greedy_construct(n_rows, k)
        if rows is None:
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                rows.append(row)
            return rows

    # Step 2: Local search to reduce scope
    if time.time() < deadline - 20:
        rows = _local_search_scope(rows, k, rng, deadline - 20)

    # Step 3: Simulated annealing to escape local optima
    if time.time() < deadline - 15:
        sa_rows = _simulated_annealing_scope(rows, k, rng, deadline - 15)
        if _total_scope(sa_rows) < _total_scope(rows):
            rows = sa_rows

    # Step 4: Multi-start local search
    if time.time() < deadline - 10:
        ms_rows = _multi_start_local_search(n_rows, k, rng, deadline - 10)
        if ms_rows is not None and _total_scope(ms_rows) < _total_scope(rows):
            rows = ms_rows

    # Step 5: Final polish
    if time.time() < deadline - 3:
        rows = _polish_pass(rows, k, deadline - 3)

    return rows

# EVOLVE-BLOCK-END
