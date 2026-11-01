import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 135.0  # 15s safety margin

    # Number of differences in one row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Get initial solution via greedy
    best_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(best_rows)

    # Lower bound for M
    M_lower = max(total_diffs_needed, k * (k + 1) // 2 + k, 20)

    # Try to find a solution with smaller M using global CSP with restarts
    for M in range(M_lower, best_M):
        if time.time() > deadline - 10:
            break

        rows = _solve_global_csp(n_rows, k, M, deadline)
        if rows is not None and _is_valid(rows, k):
            new_M = _max_entry(rows)
            if new_M < best_M:
                best_rows = rows
                best_M = new_M
                break

    return best_rows


def _solve_global_csp(n_rows, k, M, deadline):
    """
    Global CSP solver: precompute candidate rows, then use backtracking with
    forward checking and dynamic variable ordering (most-constrained first).
    Uses randomized restarts for diversity.
    """
    # Precompute all candidate rows with max entry <= M
    candidates = _generate_candidate_rows(k, M)
    if not candidates:
        return None

    # For each candidate, compute the set of differences it uses
    cand_diffs = []
    for row in candidates:
        dset = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                dset.add(row[j] - row[i])
        cand_diffs.append(dset)

    # Total differences available in [1, M]
    max_diff = M
    total_possible_diffs = max_diff  # diffs from 1 to M

    best_result = None
    max_restarts = 80

    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break

        # Randomized restart: shuffle candidate order
        indices = list(range(len(candidates)))
        random.shuffle(indices)

        rows = _backtrack_global(n_rows, k, M, candidates, cand_diffs, indices, deadline, restart)
        if rows is not None:
            best_result = rows
            break

    return best_result


def _backtrack_global(n_rows, k, M, candidates, cand_diffs, cand_order, deadline, restart_id):
    """
    Backtracking with forward checking and dynamic variable ordering.
    We assign rows one by one. At each step, pick the unassigned row slot that
    has the fewest remaining compatible candidates (most-constrained).
    """
    used_diffs = set()
    assignment = [None] * n_rows  # row_index -> candidate_index
    remaining_candidates = [set(range(len(candidates))) for _ in range(n_rows)]

    # Initially all rows can use any candidate
    for i in range(n_rows):
        remaining_candidates[i] = set(range(len(candidates)))

    def backtrack(depth):
        if time.time() > deadline - 15:
            return False

        if depth == n_rows:
            return True

        # Dynamic variable ordering: pick the unassigned row with fewest options
        min_count = float('inf')
        min_row = -1
        for r in range(n_rows):
            if assignment[r] is None:
                cnt = len(remaining_candidates[r])
                if cnt < min_count:
                    min_count = cnt
                    min_row = r
                    if cnt == 0:
                        return False

        if min_row == -1:
            return True  # all assigned

        # Try candidates for min_row in shuffled order
        cand_indices = list(remaining_candidates[min_row])
        random.shuffle(cand_indices)

        for ci in cand_indices:
            # Check if this candidate's diffs conflict with used_diffs
            cd = cand_diffs[ci]
            if cd & used_diffs:
                continue

            # Tentative assignment
            assignment[min_row] = ci
            new_used = used_diffs | cd

            # Forward checking: update remaining candidates for other unassigned rows
            valid_next = True
            affected = []
            for r in range(n_rows):
                if r == min_row or assignment[r] is not None:
                    continue
                # Remove candidates that conflict with new_used
                before = len(remaining_candidates[r])
                filtered = set()
                for cj in remaining_candidates[r]:
                    if not (cand_diffs[cj] & new_used):
                        filtered.add(cj)
                if len(filtered) < before:
                    affected.append((r, remaining_candidates[r], filtered))
                remaining_candidates[r] = filtered
                if len(filtered) == 0:
                    valid_next = False
                    break

            if valid_next:
                if backtrack(depth + 1):
                    return True

            # Undo forward checking
            for r, old_set, _ in affected:
                remaining_candidates[r] = old_set

            # Undo assignment
            assignment[min_row] = None

        return False

    if backtrack(0):
        # Build result rows
        result = [None] * n_rows
        for r in range(n_rows):
            result[r] = candidates[assignment[r]]
        return result
    return None


def _generate_candidate_rows(k, M):
    """
    Generate a diverse pool of candidate rows of k+1 increasing ints starting at 0,
    with max entry <= M. We generate:
    1. Greedy minimal row
    2. Random rows via randomized backtracking
    3. Some structured variations
    """
    rows = []
    seen = set()

    # Greedy minimal
    row, x = [0], 0
    while len(row) <= k:
        x += 1
        if x > M:
            break
        new = {x - a for a in row}
        if len(new) == len(row):
            row.append(x)
    if len(row) == k + 1 and row[-1] <= M:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            rows.append(row)

    # Randomized backtracking to generate diverse rows
    max_attempts = 200
    for _ in range(max_attempts):
        row = _random_row_backtrack(k, M)
        if row is not None:
            key = tuple(row)
            if key not in seen:
                seen.add(key)
                rows.append(row)
            if len(rows) >= 500:  # cap pool size
                break

    return rows


def _random_row_backtrack(k, M):
    """
    Generate a single random valid row of k+1 increasing ints starting at 0, max <= M.
    Uses randomized backtracking.
    """
    def backtrack(start, chosen, current_diffs):
        if len(chosen) == k:
            return [0] + chosen[:]

        remaining = k - len(chosen)
        if M - start + 1 < remaining:
            return None

        candidates = list(range(start, M + 1))
        random.shuffle(candidates)

        for v in candidates:
            new_diffs = set()
            ok = True
            for a in chosen:
                d = v - a
                if d in current_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok or len(new_diffs) != len(chosen):
                continue

            chosen.append(v)
            current_diffs |= new_diffs
            result = backtrack(v + 1, chosen, current_diffs)
            if result is not None:
                return result
            chosen.pop()
            current_diffs -= new_diffs

        return None

    return backtrack(1, [], set())


def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _max_entry(rows):
    return max(max(row) for row in rows)


def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False

        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
