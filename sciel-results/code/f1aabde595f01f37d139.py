"""Structural redesign: CSP backtracking with difference-cost heuristic and time-bounded restarts."""
import time
import random
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 130.0  # safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')

    def greedy_seed():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def get_scope(rows):
        return max(max(r) for r in rows)

    # Start with greedy solution
    seed_rows = greedy_seed()
    best_rows = seed_rows
    best_scope = get_scope(seed_rows)

    # CSP backtracking to improve
    def backtrack(rows, row_idx, used, current_scope_limit):
        nonlocal best_rows, best_scope, deadline
        if time.time() > deadline:
            return None

        if row_idx == n_rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
            return True

        # Try to construct row row_idx
        row = [0]
        return build_row(rows, row_idx, used, row, 0, current_scope_limit)

    def build_row(rows, row_idx, used, row, pos, scope_limit):
        nonlocal best_rows, best_scope, deadline
        if time.time() > deadline:
            return False

        if len(row) == k + 1:
            rows[row_idx] = row[:]
            result = backtrack(rows, row_idx + 1, used, scope_limit)
            if result:
                return True
            return False

        # Heuristic: try values that minimize the "cost" of consuming small differences
        last_val = row[-1]
        candidates = []
        # We need to pick the next value > last_val
        # To keep scope small, we should try smaller values first
        # But we also want to avoid using up small differences that are scarce
        start = last_val + 1
        max_possible = min(scope_limit, best_scope - 1) if row_idx < n_rows - 1 else best_scope - 1

        # Limit search space for performance
        for x in range(start, max_possible + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            candidates.append(x)

        # Sort candidates: prefer those that use larger differences (preserve small ones)
        # but also keep the value small to maintain tight scope
        # Use a heuristic: score = -max(new_diffs) to prefer rows that push large diffs
        # Actually, we want to save small differences for future rows, so prefer
        # candidates whose new differences are as large as possible
        def candidate_score(x):
            new_diffs = {x - a for a in row}
            # Prefer candidates that don't use very small differences
            return min(new_diffs)  # larger min diff = better (saves small diffs)

        # Try candidates in order of preference
        # For speed, limit the number of candidates we try per level
        max_candidates = 50
        if len(candidates) > max_candidates:
            # Sort by heuristic and take top candidates
            candidates.sort(key=candidate_score, reverse=True)
            candidates = candidates[:max_candidates]

        for x in candidates:
            new_diffs = {x - a for a in row}
            row.append(x)
            used |= new_diffs

            if build_row(rows, row_idx, used, row, pos + 1, scope_limit):
                return True

            used -= new_diffs
            row.pop()

        return False

    # Try to improve with backtracking, reducing scope limit each iteration
    current_limit = best_scope - 1
    max_iterations = 200

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if current_limit < 0:
            break

        # Reset random seed for reproducibility within iteration
        rows = [None] * n_rows
        used = set()

        # Build rows one by one with backtracking
        success = build_row(rows, 0, used, [0], 0, current_limit)

        if success:
            # Check if we actually improved
            if rows and all(r is not None for r in rows):
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in rows]
                # Try to improve further
                current_limit = best_scope - 1
        else:
            # If we couldn't fit within current_limit, try a slightly larger limit
            # But don't exceed best_scope
            if current_limit >= best_scope:
                break
            # Increment limit and retry
            current_limit += 1
            if current_limit >= best_scope:
                break

    return best_rows
