"""Difference triangle set: simulated annealing with greedy initialization."""
import random
import time

DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _row_diffs(row):
    """Return the set of all positive differences within a single row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the union of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _max_entry(rows):
    return max(row[-1] for row in rows)


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes smallest next mark with fresh differences."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _random_construct(n_rows, k, max_val):
    """Random construction: pick random increasing sequences within [0, max_val]."""
    rows = []
    for _ in range(n_rows):
        # Pick k+1 distinct values from 0..max_val, always include 0
        rest = random.sample(range(1, max_val + 1), k)
        row = sorted([0] + rest)
        rows.append(row)
    return rows


def _is_valid(rows, n_rows, k):
    """Check if all differences across all rows are distinct."""
    if len(rows) != n_rows:
        return False
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _try_fix(rows, n_rows, k, max_val):
    """Attempt to fix a nearly-valid configuration by adjusting individual entries."""
    # Compute all diffs and find conflicts
    diff_count = {}
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diff_count[d] = diff_count.get(d, 0) + 1

    conflicts = {d for d, c in diff_count.items() if c > 1}
    if not conflicts:
        return rows, True

    # Try to resolve conflicts by moving entries
    rows = [list(r) for r in rows]
    for _ in range(500):
        # Pick a random conflicting difference
        if not conflicts:
            break
        d = random.choice(list(conflicts))
        # Find a row that produces this difference
        found = False
        for ri, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] - row[i] == d:
                        # Try to move one of the entries
                        if random.random() < 0.5:
                            # Move entry j
                            old = row[j]
                            for new in range(old + 1, min(old + 20, max_val + 1)):
                                row[j] = new
                                if _is_valid(rows, n_rows, k):
                                    return rows, True
                            row[j] = old
                        else:
                            # Move entry i (but not 0)
                            if i > 0:
                                old = row[i]
                                for new in range(old + 1, min(old + 10, row[i + 1])):
                                    row[i] = new
                                    if _is_valid(rows, n_rows, k):
                                        return rows, True
                                row[i] = old
                        found = True
                        break
                if found:
                    break
            if found:
                break
        # Recompute conflicts
        diff_count = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
        conflicts = {d for d, c in diff_count.items() if c > 1}

    return rows, _is_valid(rows, n_rows, k)


def _simulated_annealing(n_rows, k, initial_rows, max_val, time_limit):
    """Simulated annealing to minimize the maximum entry."""
    start_time = time.time()
    rows = [list(r) for r in initial_rows]
    best_rows = [list(r) for r in rows]
    best_score = _max_entry(best_rows)

    # Start temperature
    T = max_val * 0.5
    T_min = 0.01
    cooling = 0.9995

    while T > T_min:
        if time.time() - start_time > time_limit:
            break

        # Random move: pick a row and an entry (not first), change its value
        ri = random.randrange(n_rows)
        row = rows[ri]
        ei = random.randrange(1, k + 1)  # indices 1..k
        old_val = row[ei]

        # Try a small random perturbation
        delta = random.randint(-15, 15)
        new_val = old_val + delta

        # Ensure constraints: strictly increasing, positive, within bounds
        valid_move = True
        if new_val <= row[ei - 1] or new_val >= max_val + 1:
            valid_move = False
        if ei < k and new_val >= row[ei + 1]:
            valid_move = False

        if not valid_move:
            T *= cooling
            continue

        # Compute cost: number of duplicate differences
        # Efficient: only recompute diffs for the changed row and check against others
        row[ei] = new_val

        # Check if still strictly increasing
        if any(row[i + 1] <= row[i] for i in range(k)):
            row[ei] = old_val
            T *= cooling
            continue

        # Compute all diffs and count duplicates
        diff_set = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    diff_set[d] = diff_set.get(d, 0) + 1

        num_conflicts = sum(1 for c in diff_set.values() if c > 1)
        cur_score = _max_entry(rows)

        if num_conflicts == 0:
            # Valid configuration
            if cur_score < best_score:
                best_score = cur_score
                best_rows = [list(r) for r in rows]
            # Accept
            T *= cooling
            continue

        # Invalid: accept with probability based on temperature
        # Cost = num_conflicts + cur_score (penalize conflicts heavily)
        cost = num_conflicts * 10 + cur_score
        best_cost = best_score
        delta_cost = cost - best_cost

        if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / max(T, 0.01)):
            T *= cooling
        else:
            # Reject: revert
            row[ei] = old_val
            T *= cooling

    return best_rows, best_score


def construct(n_rows, k):
    start_time = time.time()
    deadline = DEADLINE - start_time

    # Phase 1: Greedy initialization
    rows_greedy = _greedy_construct(n_rows, k)
    best_rows = rows_greedy
    best_score = _max_entry(best_rows)

    # Phase 2: Try random constructions and fix
    max_val = best_score + 20
    attempts = 0
    while time.time() - start_time < deadline * 0.3 and attempts < 50:
        attempts += 1
        rows_rand = _random_construct(n_rows, k, max_val)
        # Try to fix
        rows_fixed, valid = _try_fix(rows_rand, n_rows, k, max_val)
        if valid:
            score = _max_entry(rows_fixed)
            if score < best_score:
                best_score = score
                best_rows = rows_fixed
                max_val = score + 15

    # Phase 3: Simulated annealing starting from best
    if time.time() - start_time < deadline * 0.7:
        sa_time = deadline * 0.6 - (time.time() - start_time)
        if sa_time > 2:
            rows_sa, score_sa = _simulated_annealing(n_rows, k, best_rows, best_score + 10, sa_time)
            if score_sa < best_score:
                best_score = score_sa
                best_rows = rows_sa

    # Phase 4: Local search / hill climbing from best
    if time.time() - start_time < deadline * 0.9:
        local_time = deadline * 0.85 - (time.time() - start_time)
        if local_time > 1:
            rows_local = [list(r) for r in best_rows]
            local_start = time.time()
            improved = True
            while improved and time.time() - local_start < local_time:
                improved = False
                for ri in range(n_rows):
                    for ei in range(1, k + 1):
                        old = rows_local[ri][ei]
                        for new in range(old - 10, old + 11):
                            if new <= rows_local[ri][ei - 1]:
                                continue
                            if new >= rows_local[ri][ei + 1] if ei < k else new > best_score + 5:
                                continue
                            rows_local[ri][ei] = new
                            if _is_valid(rows_local, n_rows, k):
                                s = _max_entry(rows_local)
                                if s < best_score:
                                    best_score = s
                                    best_rows = [list(r) for r in rows_local]
                                    improved = True
                                else:
                                    rows_local[ri][ei] = old
                                break
                            rows_local[ri][ei] = old

    return best_rows
