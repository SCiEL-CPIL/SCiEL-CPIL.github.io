import time
import random
import sys

sys.setrecursionlimit(10000)


def construct(n_rows, k):
    """Construct a difference triangle set with n_rows rows of k+1 increasing
    integers starting at 0, minimizing the maximum entry."""
    deadline = time.time() + 115  # Safety margin before 120s limit

    # Generate initial solution using greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)

    # Iterated Local Search with valid perturbations
    while time.time() < deadline - 15:
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]

        # Perform several random valid moves (perturbation)
        for _ in range(random.randint(3, 8)):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Don't touch position 0

            # Find a random valid new value for candidate[r_idx][pos]
            new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val

        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline)

        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved

    # Final polish pass with remaining time
    if time.time() < deadline - 2:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution


def _row_diffs(row):
    """Return the set of all positive differences within a single row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _compute_all_diffs(rows):
    """Return the union of all positive differences across all rows."""
    all_diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                all_diffs.add(r[j] - ri)
    return all_diffs


def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """Find a random valid value for rows[r_idx][pos] different from current."""
    row = rows[r_idx]
    old_val = row[pos]

    # Build the set of differences not involving the old value at this position
    # (i.e., the differences that must remain unused after the change).
    # We compute all diffs once, then remove those involving old_val.
    all_diffs = _compute_all_diffs(rows)

    diffs_involving_old = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)

    forbidden_diffs = all_diffs - diffs_involving_old

    # Search for a new value
    lower_bound = row[pos - 1] + 1
    upper_bound = 300  # Slightly wider than before
    if pos + 1 < len(row):
        upper_bound = min(upper_bound, row[pos + 1] - 1)

    if upper_bound < lower_bound:
        return None

    # Collect valid candidates
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue

        # Check validity: each new diff must not collide with forbidden_diffs
        # or with another new diff in the same row.
        valid = True
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val

            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)

        if valid:
            valid_candidates.append(new_val)

    if not valid_candidates:
        return None

    # Pick randomly
    return random.choice(valid_candidates)


def _local_search_minimize(rows, n_rows, k, deadline):
    """Greedy local search to minimize max entry.

    Walks positions right-to-left within each row so that lowering an earlier
    position does not invalidate later positions (they are already fixed and
    their diffs are accounted for in used_diffs).
    """
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)

    while time.time() < deadline - 5:
        progress = False

        # Randomize row order for diversity
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 5:
                break

            row = best_rows[r_idx]

            # Process positions right-to-left
            for pos in range(k, 0, -1):
                old_val = row[pos]

                # Compute forbidden diffs: all used diffs minus those involving old_val
                diffs_involving_old = set()
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)

                forbidden_diffs = used_diffs - diffs_involving_old

                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < len(row):
                    upper_bound = min(upper_bound, row[pos + 1] - 1)

                if upper_bound < lower_bound:
                    continue

                # Search for smallest valid value
                found = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val

                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                    if valid:
                        found = new_val
                        break

                if found is not None and found < old_val:
                    # Update used_diffs incrementally
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True

        if not progress:
            break

    return best_rows


def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows
