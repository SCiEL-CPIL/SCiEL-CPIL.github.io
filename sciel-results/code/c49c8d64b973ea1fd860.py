import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a constraint-satisfaction approach: row-by-row construction with
    conflict-driven row repair (row swap + local search on the repaired row).
    """
    start_time = time.time()
    deadline = start_time + 135.0

    # Lower bound on scope
    total_diffs_needed = n_rows * k * (k + 1) // 2

    # ---- Greedy fallback ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _global_max(rows):
        return max(max(row) for row in rows)

    def _build_diffs_index(rows):
        """Build a dict: diff_value -> list of row indices containing that diff."""
        index = {}
        for ri, row in enumerate(rows):
            for d in _row_diffs(row):
                if d not in index:
                    index[d] = []
                index[d].append(ri)
        return index

    def _resolve_row_given_others(others_diffs, k, max_scope, rng):
        """
        Find a valid row of k+1 increasing integers starting at 0
        that avoids all differences in others_diffs, with max <= max_scope.
        Uses randomized backtracking with local search.
        """
        n = k + 1
        other_set = set(others_diffs)

        def _try_build_partial(used_local, row, pos):
            """Recursively build the row position by position."""
            if pos == n:
                return row[:]
            prev = row[pos - 1] if pos > 0 else -1
            # Upper bound for this position
            if pos < n - 1:
                # Leave room for remaining elements
                ub = max_scope - (n - 1 - pos)
            else:
                ub = max_scope
            # Try values in random-ish order, starting from low
            start_val = prev + 1
            # Collect candidates
            candidates = list(range(start_val, ub + 1))
            rng.shuffle(candidates)
            # Try fewer candidates first for speed
            for v in candidates[:200]:
                new_diffs = {v - a for a in row}
                if new_diffs & other_set:
                    continue
                if new_diffs & used_local:
                    continue
                new_used = used_local | new_diffs
                result = _try_build_partial(new_used, row + [v], pos + 1)
                if result is not None:
                    return result
            return None

        # Multiple random attempts
        for attempt in range(30):
            rng2 = random.Random(attempt * 31 + n_rows * 7)
            result = _try_build_partial(set(), [0], 1)
            if result is not None:
                return result
        return None

    def _local_search_row(row, other_diffs, k, max_scope, rng, iterations=200):
        """
        Local search to minimize max entry of a single row,
        given fixed differences from other rows.
        """
        n = k + 1
        other_set = set(other_diffs)
        best_row = list(row)
        best_max = max(best_row)

        for _ in range(iterations):
            # Pick a position to modify (not 0)
            pi = rng.randint(1, n - 1)
            old_val = best_row[pi]
            # Try decreasing first
            min_allowed = best_row[pi - 1] + 1
            max_allowed = (best_row[pi + 1] - 1) if pi < n - 1 else max_scope - 1

            candidates = []
            # Decreasing candidates
            for d in range(1, min(old_val - min_allowed + 1, 50)):
                candidates.append(old_val - d)
            # Some increasing for diversity
            for d in range(1, min(max_allowed - old_val + 1, 20)):
                candidates.append(old_val + d)
            rng.shuffle(candidates)

            improved = False
            for new_val in candidates:
                if new_val <= best_row[pi - 1]:
                    continue
                if pi < n - 1 and new_val >= best_row[pi + 1]:
                    continue
                cand = best_row[:]
                cand[pi] = new_val
                # Check internal validity
                diffs = set()
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        d = cand[j] - cand[i]
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                if not valid:
                    continue
                # Check against other rows
                if diffs & other_set:
                    continue
                new_max = max(cand)
                if new_max < best_max:
                    best_row = cand
                    best_max = new_max
                    improved = True
                    break
                elif new_max == best_max and rng.random() < 0.1:
                    best_row = cand
                    improved = True
                    break
            if not improved:
                break

        return best_row

    def _conflict_matrix(rows):
        """
        For each row, count how many of its differences conflict with other rows.
        Returns list of conflict counts.
        """
        all_diffs_index = _build_diffs_index(rows)
        n = k + 1
        conflicts = []
        for ri in range(len(rows)):
            row_diffs = _row_diffs(rows[ri])
            conf = 0
            for d in row_diffs:
                owners = all_diffs_index.get(d, [])
                if len(owners) > 1:
                    conf += 1
            conflicts.append(conf)
        return conflicts

    def _repair_and_swap(rows, ri, others_diffs, max_scope, rng):
        """
        Try to replace row ri with a new valid row, then locally optimize.
        Returns the new row or None.
        """
        # First try to find a fresh row
        new_row = _resolve_row_given_others(others_diffs, k, max_scope, rng)
        if new_row is None:
            return None
        # Local search to reduce its max
        new_row = _local_search_row(new_row, others_diffs, k, max_scope, rng, iterations=100)
        return new_row

    # ---- Main search: iterative conflict-driven repair ----
    def search_with_conflict_repair(n_rows, k, start_time, deadline):
        rng = random.Random(hash((n_rows, k, start_time)) % 10**8)

        # Start from greedy
        rows = greedy_construct(n_rows, k)
        current_scope = _global_max(rows)

        # Track best
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        no_improve = 0
        max_no_improve = 5000
        restarts = 0
        max_restarts = 15

        while time.time() < deadline:
            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                rows = [r[:] for r in best_rows]
                current_scope = best_scope
                no_improve = 0
                # Perturb: randomly pick 2 rows and swap their contents
                if n_rows >= 2:
                    i, j = rng.sample(range(n_rows), 2)
                    rows[i], rows[j] = rows[j], rows[i]

            # Find rows with most conflicts
            all_diffs_index = _build_diffs_index(rows)
            conflict_counts = []
            for ri in range(n_rows):
                row_diffs = _row_diffs(rows[ri])
                conf = 0
                for d in row_diffs:
                    owners = all_diffs_index.get(d, [])
                    if len(owners) > 1:
                        conf += 1
                conflict_counts.append(conf)

            # Pick the most conflicted row
            max_conf = max(conflict_counts)
            if max_conf == 0:
                # No conflicts! Check if we can reduce scope
                # Try to reduce each row
                improved_any = False
                for ri in range(n_rows):
                    if time.time() > deadline:
                        break
                    others_diffs = set()
                    for j in range(n_rows):
                        if j != ri:
                            others_diffs |= _row_diffs(rows[j])
                    # Try to reduce this row's max
                    new_row = _local_search_row(rows[ri], others_diffs, k, best_scope, rng, iterations=50)
                    if max(new_row) < max(rows[ri]):
                        rows[ri] = new_row
                        current_scope = _global_max(rows)
                        improved_any = True
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                            no_improve = 0
                        else:
                            no_improve += 1
                        break
                if not improved_any:
                    no_improve += 1
                    # Random perturbation
                    ri = rng.randint(0, n_rows - 1)
                    pi = rng.randint(1, k)
                    old_val = rows[ri][pi]
                    min_allowed = rows[ri][pi - 1] + 1
                    max_allowed = (rows[ri][pi + 1] - 1) if pi < k else current_scope
                    new_val = rng.randint(min_allowed, max_allowed)
                    if new_val != old_val:
                        cand = rows[ri][:]
                        cand[pi] = new_val
                        # Validate
                        diffs = set()
                        valid = True
                        n = len(cand)
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = cand[j] - cand[i]
                                if d in diffs:
                                    valid = False
                                    break
                                diffs.add(d)
                        if valid:
                            # Check cross-row
                            for d in diffs:
                                owners = all_diffs_index.get(d, [])
                                if owners and ri not in owners:
                                    valid = False
                                    break
                                elif owners and ri in owners:
                                    # This diff was already in this row, check if it's still valid
                                    pass
                        if valid:
                            rows[ri] = cand
                            current_scope = _global_max(rows)
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in rows]
                                no_improve = 0
                            else:
                                no_improve += 1
                continue

            # Pick most conflicted row
            conflicted_rows = [i for i in range(n_rows) if conflict_counts[i] == max_conf]
            ri = rng.choice(conflicted_rows)

            # Build others' diffs
            others_diffs = set()
            for j in range(n_rows):
                if j != ri:
                    others_diffs |= _row_diffs(rows[j])

            # Try to repair row ri
            new_row = _repair_and_swap(rows, ri, others_diffs, best_scope, rng)
            if new_row is not None:
                rows[ri] = new_row
                current_scope = _global_max(rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Try swapping row ri with a random other row, then repair
                if n_rows >= 2:
                    j = rng.choice([x for x in range(n_rows) if x != ri])
                    rows[ri], rows[j] = rows[j], rows[ri]
                    current_scope = _global_max(rows)
                    no_improve += 1
                    # Try to repair the new row ri
                    others_diffs2 = set()
                    for jj in range(n_rows):
                        if jj != ri:
                            others_diffs2 |= _row_diffs(rows[jj])
                    new_row2 = _repair_and_swap(rows, ri, others_diffs2, best_scope, rng)
                    if new_row2 is not None:
                        rows[ri] = new_row2
                        current_scope = _global_max(rows)
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                            no_improve = 0

            # Occasionally do a full local search pass
            if no_improve % 20 == 19:
                for ri2 in range(n_rows):
                    if time.time() > deadline:
                        break
                    others_diffs3 = set()
                    for j in range(n_rows):
                        if j != ri2:
                            others_diffs3 |= _row_diffs(rows[j])
                    new_row3 = _local_search_row(rows[ri2], others_diffs3, k, best_scope, rng, iterations=30)
                    if max(new_row3) < max(rows[ri2]):
                        rows[ri2] = new_row3
                        current_scope = _global_max(rows)
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]

        return best_rows, best_scope

    # Run search
    sa_rows, sa_scope = search_with_conflict_repair(n_rows, k, start_time, deadline)

    if sa_scope < best_scope:
        best_scope = sa_scope
        best_solution = sa_rows

    # Final polish
    if time.time() < deadline - 2:
        rng = random.Random(42)
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            improved = False
            for ri in range(n_rows):
                if time.time() > deadline - 2:
                    break
                others_diffs = set()
                for j in range(n_rows):
                    if j != ri:
                        others_diffs |= _row_diffs(best_solution[j])
                new_row = _local_search_row(best_solution[ri], others_diffs, k, best_scope, rng, iterations=100)
                if max(new_row) < max(best_solution[ri]):
                    best_solution[ri] = new_row
                    best_scope = _global_max(best_solution)
                    improved = True
            if not improved:
                break

    # Final validation
    if not _validate(best_solution, n_rows, k):
        # Fallback to greedy
        best_solution = greedy_construct(n_rows, k)

    return best_solution
# EVOLVE-BLOCK-END
