import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    
    Approach: CSP with resource partitioning of differences.
    - The set of differences {1, 2, ..., S-1} is a shared resource.
    - Each row consumes k*(k+1)/2 distinct differences.
    - Backtracking selects next element by degree heuristic:
      choose the candidate that maximizes the number of remaining valid
      difference slots for subsequent positions.
    """
    start_time = time.time()
    time_limit = 115.0

    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Lower bound on scope
    lb = total_diffs_needed + 1

    best_result = None
    best_scope = float('inf')

    def count_remaining_slots(used_diffs, current_row, row_idx, S):
        """
        Count how many future (row, position) pairs can still be filled.
        A slot is feasible if there exists at least one value that doesn't
        conflict with used_diffs.
        Returns the count of feasible slots.
        """
        remaining_slots = 0

        # Positions remaining in current row
        for pos in range(len(current_row), k + 1):
            # For position pos, we need a value x > last element (or x > 0 if pos==0, but pos>=1 here since current_row has at least [0])
            last = current_row[-1] if current_row else 0
            # Find if there's any valid x
            found = False
            # Upper bound: must allow enough room for remaining positions
            rem_after = k + 1 - pos - 1  # positions after this one
            max_x = S - 1 - rem_after
            for x in range(last + 1, max_x + 1):
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & used_diffs):
                    found = True
                    break
            if found:
                remaining_slots += 1
            else:
                # No valid value for this position, so all subsequent positions in this row are infeasible
                break

        # Positions in future rows
        for r in range(row_idx + 1, n_rows):
            # Each future row starts with [0], then needs k more elements
            # For a fresh row, position 0 is 0 (no diffs), positions 1..k need values
            # Check if at least the first few positions are feasible
            row = [0]
            feasible = True
            for pos in range(1, k + 1):
                last = row[-1]
                rem_after = k + 1 - pos - 1
                max_x = S - 1 - rem_after
                found = False
                for x in range(last + 1, max_x + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        found = True
                        break
                if not found:
                    feasible = False
                    break
                row.append(x)
            if feasible:
                remaining_slots += (k + 1)  # All positions in this row are feasible
            # Note: this is an overestimate since we don't track per-position, but good enough for heuristic

        return remaining_slots

    def solve_with_scope(S):
        """
        Try to find a solution with all entries < S (max entry <= S-1).
        """
        nonlocal best_result, best_scope, start_time, time_limit

        used_diffs = set()
        rows = []

        def _backtrack(row_idx, current_row):
            nonlocal used_diffs, rows

            if time.time() - start_time > time_limit:
                return False

            if row_idx == n_rows:
                scope = max(r[-1] for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_result = [r[:] for r in rows]
                return True

            if len(current_row) == k + 1:
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False

            last_val = current_row[-1]
            pos_in_row = len(current_row)

            # Generate candidates with degree heuristic
            candidates = []

            # Max x: must leave room for remaining positions
            rem_positions = k + 1 - pos_in_row - 1
            max_x = S - 1 - rem_positions

            for x in range(last_val + 1, max_x + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue

                # Degree: count remaining feasible slots after placing x
                new_used = used_diffs | new_diffs
                degree = count_remaining_slots(new_used, current_row + [x], row_idx, S)

                candidates.append((degree, x, new_diffs))

            if not candidates:
                return False

            # Sort: highest degree first (most flexible), then smallest x (tightest scope)
            candidates.sort(key=lambda item: (-item[0], item[1]))

            # Try top candidates (limit to avoid excessive branching)
            limit = min(len(candidates), 20)
            for i in range(limit):
                deg, x, new_diffs = candidates[i]
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs

                if _backtrack(row_idx, current_row + [x]):
                    return True

                used_diffs = old_used

            return False

        return _backtrack(0, [0])

    # Iterative deepening: try scopes from lb upward
    # Start with a reasonable range
    max_search = lb + 80

    for S in range(lb, max_search + 1):
        if time.time() - start_time > time_limit - 3:
            break

        if solve_with_scope(S):
            # Found solution; break since we're going low to high
            break

    if best_result is None:
        # Fallback: greedy
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

    return best_result
