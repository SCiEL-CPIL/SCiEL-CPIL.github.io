"""
Difference triangle set (7,5) solver.

Strategy (structural change):
- Model the problem as a constraint satisfaction search over the global
  set of required differences.
- Each row is represented by its 15 distinct positive differences
  (for k=5, a row has 6 elements -> 15 pairwise differences).
- A backtracking algorithm assigns difference sets to rows, using a
  scarcity-aware cost heuristic that prefers rows that consume less
  "scarce" small differences, preserving flexibility for later rows.
- Realizability of a difference set is verified by checking that the
  differences can be embedded into a strictly increasing sequence of
  k+1 integers starting at 0 (via a small backtracking that assigns
  positions).
- After finding a valid set of n_rows, a local-search polish reduces
  the scope (maximum entry).
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START


def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= set(_diffs_of_row(r))
    return used


def _is_realizable(diff_set, k, max_scope):
    """
    Check whether a set of exactly C(k+1,2) positive integers can be
    realized as the pairwise differences of a strictly increasing
    sequence of k+1 integers starting at 0, with max entry <= max_scope.
    Returns the sequence if realizable, else None.
    """
    n = k + 1
    expected = n * (n - 1) // 2
    if len(diff_set) != expected:
        return None

    # The largest difference must equal the last element (since a[0]=0).
    max_d = max(diff_set)
    if max_d > max_scope:
        return None

    # The set must contain a[1], a[2], ..., a[n-1] (differences from 0).
    # And for each i<j, a[j]-a[i] must be in the set.
    # We backtrack over a[1..n-1].

    diffs_remaining = set(diff_set)
    # a[0] = 0
    a = [0]

    def backtrack(pos):
        """Assign a[pos] for pos in 1..n-1."""
        if pos == n:
            return a[:]
        # a[pos] must be in diffs_remaining (it's a[pos]-a[0])
        # and must be > a[pos-1]
        # Try candidates in increasing order
        for cand in sorted(diffs_remaining):
            if cand <= a[pos - 1]:
                continue
            # Check: for all i < pos, cand - a[i] must be in diffs_remaining
            ok = True
            for i in range(pos):
                d = cand - a[i]
                if d not in diffs_remaining:
                    ok = False
                    break
            if not ok:
                continue
            # Also cand itself must be in diffs_remaining
            if cand not in diffs_remaining:
                continue
            a.append(cand)
            # Remove used differences: cand - a[i] for i in 0..pos-1
            # (cand - a[pos-1] is the smallest, cand - a[0] = cand is largest)
            for i in range(pos):
                diffs_remaining.discard(cand - a[i])
            result = backtrack(pos + 1)
            if result is not None:
                return result
            # Undo
            a.pop()
            for i in range(pos):
                diffs_remaining.add(cand - a[i])
        return None

    return backtrack(1)


def _row_from_diffs(diff_set, k):
    """
    Given a set of C(k+1,2) differences, find a strictly increasing
    sequence of k+1 integers starting at 0 that realizes them.
    Returns the sequence or None.
    """
    n = k + 1
    expected = n * (n - 1) // 2
    if len(diff_set) != expected:
        return None

    a = [0]
    diffs_remaining = set(diff_set)

    def backtrack(pos):
        if pos == n:
            return a[:]
        for cand in sorted(diffs_remaining):
            if cand <= a[pos - 1]:
                continue
            if cand not in diffs_remaining:
                continue
            ok = True
            for i in range(pos):
                d = cand - a[i]
                if d not in diffs_remaining:
                    ok = False
                    break
            if not ok:
                continue
            a.append(cand)
            for i in range(pos):
                diffs_remaining.discard(cand - a[i])
            result = backtrack(pos + 1)
            if result is not None:
                return result
            a.pop()
            for i in range(pos):
                diffs_remaining.add(cand - a[i])
        return None

    return backtrack(1)


def _scarcity_weight(d, max_scope):
    """
    Heuristic weight for a difference value d.
    Smaller differences are more "scarce" because they are needed
    by more potential rows. We use a convex penalty.
    """
    # Differences near 1 are most scarce; differences near max_scope are least.
    # Use a simple inverse-ish weighting.
    return 1.0 / (1.0 + d * 0.05)


def _row_cost(diff_set, used, max_scope):
    """
    Cost of adding a row with the given difference set, given the
    already-used differences. Lower cost is better.
    The cost reflects how much scarce difference capacity is consumed.
    """
    cost = 0.0
    for d in diff_set:
        # If d is already used, this row is invalid (but we filter those out).
        # Weight by scarcity: smaller d costs more.
        cost += _scarcity_weight(d, max_scope)
    # Bonus for using larger differences (less scarce)
    return cost


def _generate_candidate_rows_csp(k, max_scope, n_candidates, rng):
    """
    Generate candidate rows by randomly selecting valid difference sets
    and checking realizability. We generate by picking k+1 elements
    directly (random increasing sequence) and computing their differences.
    This ensures realizability by construction.
    """
    candidates = []
    seen = set()
    n = k + 1
    for _ in range(n_candidates):
        # Generate a random increasing sequence of n elements starting at 0
        # with max <= max_scope
        row = [0]
        x = 0
        for _ in range(n - 1):
            # Pick next element: bias toward smaller values for tighter packing
            # but allow some spread
            if x + 1 >= max_scope:
                break
            # Random gap between 1 and some max
            max_gap = min(15, max_scope - x)
            gap = rng.randint(1, max_gap)
            x += gap
            if x > max_scope:
                x = max_scope
            row.append(x)
        if len(row) < n:
            continue
        diffs = _diffs_of_row(row)
        if diffs in seen:
            continue
        seen.add(diffs)
        candidates.append((row, diffs))
    return candidates


def _csp_backtracking(n_rows, k, max_scope, deadline, rng):
    """
    CSP backtracking: assign rows one at a time, each row being a
    specific increasing sequence. The constraint is that all differences
    across rows are distinct.
    
    We use a pool of pre-generated candidate rows and select a
    compatible subset of size n_rows, minimizing scope.
    
    The key structural difference from the old approach: we use a
    scarcity-aware ordering of candidate rows (prefer rows that use
    less scarce differences) and a forward-checking style pruning.
    """
    # Generate a diverse pool of candidate rows
    pool = _generate_candidate_rows_csp(k, max_scope, 500, rng)
    
    if len(pool) < n_rows:
        return None

    # Sort candidates by cost (scarcity-aware): prefer rows that use
    # larger differences (less scarce) so that small differences remain
    # available for rows that need them.
    # Actually, we want to be careful: we want to find a set of n_rows
    # mutually compatible rows. The ordering affects which solutions
    # we find first.
    
    # Compute a "scarcity cost" for each candidate: sum of weights of its diffs
    # Rows with lower total scarcity cost are "cheaper" to place.
    # But we also want diversity. Let's sort by max entry (scope contribution)
    # and then by scarcity cost.
    
    def candidate_sort_key(item):
        row, diffs = item
        scope_contrib = row[-1]
        scarcity = sum(_scarcity_weight(d, max_scope) for d in diffs)
        return (scope_contrib, scarcity)
    
    pool_sorted = sorted(pool, key=candidate_sort_key)
    
    # Backtracking search for n_rows mutually compatible rows
    best_rows = [None]
    best_scope = [max_scope + 1]
    node_count = [0]
    
    def backtrack(idx, used_diffs, selected):
        nonlocal node_count
        if time.time() > deadline:
            return
        node_count[0] += 1
        
        if len(selected) == n_rows:
            scope = max(row[-1] for row, _ in selected)
            if scope < best_scope[0]:
                best_scope[0] = scope
                best_rows[0] = [row[:] for row, _ in selected]
            return
        
        # Upper bound: even if we pick the smallest-scope remaining candidates,
        # can we beat the current best?
        remaining_needed = n_rows - len(selected)
        if remaining_needed > len(pool_sorted) - idx:
            return
        
        # Try candidates starting from idx
        for i in range(idx, len(pool_sorted)):
            if time.time() > deadline:
                return
            row, diffs = pool_sorted[i]
            # Prune: if this row's scope already exceeds best, skip
            # (but other rows might be smaller, so only skip if ALL
            #  remaining would exceed)
            if row[-1] >= best_scope[0]:
                # Since pool is sorted by scope, all subsequent are >= this
                break
            # Check compatibility
            if diffs & used_diffs:
                continue
            # Forward check: ensure there are enough remaining candidates
            # that are compatible with (used_diffs | diffs)
            # (lightweight: just count)
            selected.append((row, diffs))
            new_used = used_diffs | diffs
            backtrack(i + 1, new_used, selected)
            selected.pop()
            if best_rows[0] is not None and best_scope[0] <= row[-1]:
                # Can't improve further with larger-scope rows
                break
    
    backtrack(0, set(), [])
    
    if best_rows[0] is not None:
        return best_rows[0]
    return None


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    scope = _total_scope(rows)
    
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
            
            # Try to reduce the last element
            last = row[-1]
            for new_last in range(last - 1, row[-2], -1):
                old_diffs = {last - a for a in row[:-1]}
                new_diffs = {new_last - a for a in row[:-1]}
                remaining = used - old_diffs
                if not (new_diffs & remaining):
                    row[-1] = new_last
                    used = (used - old_diffs) | new_diffs
                    scope = _total_scope(rows)
                    improved = True
                    break
            
            # Try to reduce other elements
            for pos in range(len(row) - 2, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
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
                        scope = _total_scope(rows)
                        improved = True
                        break
    
    return rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    best_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    
    for _ in range(10):
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
                    for new_val in range(old_val - 1, row[pos - 1], -1):
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
        
        current_scope = _total_scope(rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in rows]
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
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


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Get a baseline via greedy
    rows_greedy = _greedy_construct(n_rows, k)
    if rows_greedy is None:
        return []
    best_rows = rows_greedy
    best_scope = _total_scope(best_rows)

    # CSP backtracking search with increasing scope limits
    # Start from a reasonable lower bound and work up
    # The counting bound suggests ~105, best known is 112.
    # Try scopes from a low value upward.
    
    # First, try the greedy scope as an upper bound
    greedy_scope = best_scope
    
    # Try CSP search with various scope limits
    # We want to find the minimum scope that allows n_rows compatible rows.
    # Binary search or linear search over scope.
    
    # Lower bound: at least we need enough distinct differences.
    # Each row uses C(k+1,2) differences. n_rows rows need n_rows * C(k+1,2)
    # distinct differences, all <= scope. So scope >= n_rows * C(k+1,2).
    # For n_rows=7, k=5: 7*15 = 105.
    lower_bound = n_rows * (k + 1) * k // 2
    
    # Try scopes from lower_bound to greedy_scope
    # Use a step to avoid too many iterations
    # Actually, try a few key values: lower_bound, lower_bound+5, ..., greedy_scope
    
    scopes_to_try = []
    # Include lower bound and a few values above it
    for s in range(lower_bound, min(greedy_scope + 1, lower_bound + 40), 2):
        scopes_to_try.append(s)
    if greedy_scope not in scopes_to_try:
        scopes_to_try.append(greedy_scope)
    # Also try some values above greedy_scope in case greedy is suboptimal
    # (unlikely but safe)
    scopes_to_try = sorted(set(scopes_to_try))
    
    found_csp = False
    for scope_limit in scopes_to_try:
        if time.time() > deadline - 20:
            break
        result = _csp_backtracking(n_rows, k, scope_limit, deadline - 20, rng)
        if result is not None:
            result_scope = _total_scope(result)
            if result_scope < best_scope:
                best_rows = result
                best_scope = result_scope
            found_csp = True
            # If we found a solution at the lower bound, we're likely optimal
            # (or near-optimal). Try to improve with local search.
            if scope_limit <= lower_bound + 10:
                break
    
    # Local search polish
    if time.time() < deadline - 10:
        best_rows = _local_search_scope(best_rows, k, rng, deadline - 10)
    
    # High-precision polish
    if time.time() < deadline - 3:
        best_rows = _polish_pass(best_rows, k, deadline - 3)
    
    return best_rows

# EVOLVE-BLOCK-END
