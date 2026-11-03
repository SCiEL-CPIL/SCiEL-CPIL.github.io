"""Difference triangle set (4,7): CSP over difference values with bipartite matching."""
import random
import sys
import time
from collections import defaultdict

sys.setrecursionlimit(100000)


def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Uses a CSP formulation over difference values with bipartite matching for assignment.
    """

    def greedy():
        """Greedy baseline: construct rows sequentially, picking smallest valid values."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used.add(d)
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)

    def check_valid(rows):
        """Verify that all positive differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def solve_csp(scope, seed, time_limit=100.0):
        """
        CSP solver: assign difference values to slots such that each row's differences
        form a valid additive structure.
        
        For a row [a0=0, a1, ..., ak], the differences are:
        - Direct: a_i - a_j for i > j
        - These must all be distinct globally
        
        We formulate this as: for each row, choose k+1 values from [0, scope]
        such that all pairwise differences are globally unique.
        
        We use backtracking with forward checking on the difference set.
        """
        rng = random.Random(seed)
        t_start = time.time()
        used_diffs = set()
        rows = []
        node_count = [0]
        max_nodes = 500000

        def can_place(row, x, used_local):
            """Check if x can be added to row without creating duplicate differences."""
            for a in row:
                d = x - a
                if d in used_local or d in used_diffs:
                    return False
            return True

        def backtrack(row_idx, min_start):
            nonlocal used_diffs, node_count
            if time.time() - t_start > time_limit:
                return None
            if node_count[0] > max_nodes:
                return None
            node_count[0] += 1

            if row_idx == n_rows:
                return list(rows)

            # Symmetry breaking: rows sorted by their second element
            if row_idx == 0:
                lo = 1
            else:
                lo = rows[row_idx - 1][1] + 1

            row = [0]
            row_diffs = set()

            def rec_row(row, row_diffs, min_x):
                nonlocal used_diffs, node_count
                if time.time() - t_start > time_limit:
                    return None
                if node_count[0] > max_nodes:
                    return None
                node_count[0] += 1

                if len(row) == k + 1:
                    # Check no conflicts with global used_diffs
                    if row_diffs & used_diffs:
                        return None
                    # Place this row
                    used_diffs.update(row_diffs)
                    rows.append(row)
                    result = backtrack(row_idx + 1, min_x)
                    if result is not None:
                        return result
                    # Backtrack
                    rows.pop()
                    used_diffs.difference_update(row_diffs)
                    return None

                remaining = (k + 1) - len(row)
                max_x = scope - (remaining - 1)

                if max_x < min_x:
                    return None

                # Generate candidates with pruning
                candidates = []
                for x in range(min_x, max_x + 1):
                    ok = True
                    for a in row:
                        d = x - a
                        if d in row_diffs or d in used_diffs:
                            ok = False
                            break
                    if ok:
                        candidates.append(x)

                if not candidates:
                    return None

                # Heuristic ordering: try smaller values first (tighter packing)
                # but also consider diversity via seed
                if seed % 4 == 0:
                    # Greedy: smallest first
                    order = candidates
                elif seed % 4 == 1:
                    # Random from first half
                    subset = candidates[:max(1, len(candidates) // 2)]
                    order = [rng.choice(subset)] + [c for c in candidates if c != subset[0]]
                elif seed % 4 == 2:
                    # Middle-biased
                    mid = len(candidates) // 2
                    order = [candidates[mid]] + candidates[:mid] + candidates[mid + 1:]
                else:
                    # Random permutation
                    order = candidates[:]
                    rng.shuffle(order)

                for x in order:
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        new_diffs.add(d)

                    if rec_row(row + [x], row_diffs | new_diffs, x + 1):
                        return True
                    return None

            return rec_row(row, row_diffs, lo)

        return backtrack(0, 1)

    t0 = time.time()
    time_limit = 115.0
    best_found = best_rows
    best_scope_found = best_scope

    # Try to improve by searching for smaller scopes
    for scope in range(best_scope - 1, max(80, best_scope - 30), -1):
        if time.time() - t0 > time_limit:
            break

        found = None
        # Try multiple seeds with different strategies
        num_seeds = 8
        for seed in range(num_seeds):
            if time.time() - t0 > time_limit:
                break
            # Allocate remaining time budget
            remaining_time = time_limit - (time.time() - t0)
            per_seed_time = remaining_time / (num_seeds - seed)
            res = solve_csp(scope, seed, time_limit=per_seed_time)
            if res is not None:
                # Verify the solution
                if check_valid(res):
                    found = res
                    break

        if found:
            best_found = found
            best_scope_found = scope
        else:
            # If we can't find a solution at this scope, try the next smaller one
            # but only if we have time
            if time.time() - t0 > time_limit - 20:
                break

    return best_found
