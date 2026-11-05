import random
import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_ok():
        return elapsed() < time_limit

    def row_diffs(row):
        """Return the set of all positive differences in a row."""
        n = len(row)
        return {row[j] - row[i] for i in range(n) for j in range(i + 1, n)}

    def expected_diff_count(n):
        return n * (n - 1) // 2

    # Precompute: for a row of length k+1, we need k+1 positions (0..k),
    # position 0 is always 0, positions 1..k are to be filled.
    # Each position p (1-indexed from 1 to k) must be > position p-1.
    # The differences contributed by adding value x at position p are {x - row[i] for i in 0..p-1}.

    def solve_csp(rng, scope_cap):
        """
        CSP backtracking with MRV and forward checking.
        We build rows one at a time; within a row, positions 1..k.
        Global state: used_diffs (set of all differences used so far).
        Returns list of rows or None if failed within time.
        """
        used_diffs = set()
        rows = []
        current_row = [0]

        # We'll use a recursive backtracking with explicit stack for speed,
        # but for clarity and since k is small (6), recursion is fine.
        # State: (row_index, pos_in_row, current_row, used_diffs)
        # To avoid deep recursion issues, we use iterative approach with explicit stack.

        # Actually, let's use a cleaner recursive approach with memoization-free backtracking.
        # The search space is manageable for k=6, n_rows=7.

        def backtrack(row_idx, pos, cur_row, used):
            """
            row_idx: which row we're building (0-indexed)
            pos: current position in cur_row (we're about to fill position pos)
            cur_row: the row being built (list)
            used: set of used differences
            Returns True if solution found (updates rows), False otherwise.
            """
            if not time_ok():
                return False

            if row_idx == n_rows:
                # All rows built successfully
                return True

            if pos == k + 1:
                # Current row is complete
                rows.append(list(cur_row))
                return backtrack(row_idx + 1, 1, [0], used)

            # We need to choose a value for cur_row[pos]
            # Constraints:
            #   1. cur_row[pos] > cur_row[pos-1]
            #   2. cur_row[pos] <= scope_cap
            #   3. For all i in 0..pos-1: (cur_row[pos] - cur_row[i]) not in used
            #   4. The new differences must be distinct from each other (guaranteed if cur_row is strictly increasing)

            min_val = cur_row[pos - 1] + 1
            max_val = scope_cap

            # Generate candidates in increasing order (prefer small values to keep scope tight)
            # But also try some randomization for diversity
            candidates = []
            for x in range(min_val, max_val + 1):
                new_diffs = {x - cur_row[i] for i in range(pos)}
                if new_diffs & used:
                    continue
                # Check that new_diffs are all distinct (they are, since cur_row is strictly increasing)
                candidates.append(x)

            if not candidates:
                return False

            # MRV-like: we're already at a specific position, so just try candidates.
            # To add diversity, shuffle the candidate list but keep a bias toward smaller values.
            # Strategy: with some probability, pick from the first few; otherwise random.
            if rng.random() < 0.7:
                # Try in increasing order (greedy-ish)
                ordered = candidates
            else:
                # Shuffle for diversity
                ordered = list(candidates)
                rng.shuffle(ordered)
                # But still bias: put smaller values earlier with some probability
                # Actually, let's just shuffle fully for the random branch

            for x in ordered:
                if not time_ok():
                    return False

                new_used = used | {x - cur_row[i] for i in range(pos)}
                new_row = cur_row + [x]

                if backtrack(row_idx, pos + 1, new_row, new_used):
                    return True

            return False

        # Start backtracking
        if backtrack(0, 1, [0], used_diffs):
            return rows
        return None

    def local_search(rows, max_iterations=5000):
        """
        Local search: try to reduce the maximum entry by adjusting individual positions.
        Returns improved rows and the new max scope.
        """
        if not rows:
            return rows, float('inf')

        current_max = max(max(r) for r in rows)
        n = len(rows)

        # Precompute all differences per row
        row_diff_sets = [row_diffs(r) for r in rows]
        all_diffs = set()
        for ds in row_diff_sets:
            all_diffs.update(ds)

        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time_ok():
            improved = False
            iteration += 1

            # Shuffle row order
            row_order = list(range(n))
            random.shuffle(row_order)

            for i in row_order:
                if not time_ok():
                    break

                row = rows[i]
                # Try to improve each position from last to first
                for pos in range(len(row) - 1, 0, -1):
                    if not time_ok():
                        break

                    current_val = row[pos]
                    min_val = row[pos - 1] + 1
                    max_val = current_val - 1  # We want to reduce

                    if max_val < min_val:
                        continue

                    # Differences from other rows
                    other_diffs = set()
                    for r_idx, r in enumerate(rows):
                        if r_idx != i:
                            other_diffs.update(row_diffs(r))

                    found = False
                    for v in range(min_val, max_val + 1):
                        if not time_ok():
                            break

                        candidate_row = list(row)
                        candidate_row[pos] = v

                        # Check strictly increasing
                        valid = True
                        for j in range(1, len(candidate_row)):
                            if candidate_row[j] <= candidate_row[j - 1]:
                                valid = False
                                break

                        if not valid:
                            continue

                        # Check internal differences are distinct
                        cand_diffs = row_diffs(candidate_row)
                        if len(cand_diffs) != expected_diff_count(len(candidate_row)):
                            continue

                        # Check no conflict with other rows
                        if cand_diffs & other_diffs:
                            continue

                        # Accept the change
                        rows[i] = candidate_row
                        new_max = max(max(r) for r in rows)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                        found = True
                        break

                    if found:
                        break

        return rows, current_max

    def validate(rows):
        """Validate that the solution is correct."""
        if rows is None:
            return False
        if len(rows) != n_rows:
            return False
        all_diffs = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(1, len(r)):
                if r[i] <= r[i - 1]:
                    return False
            diffs = row_diffs(r)
            if len(diffs) != expected_diff_count(k + 1):
                return False
            if all_diffs & diffs:
                return False
            all_diffs.update(diffs)
        return True

    # Main search loop
    best_rows = None
    best_score = float('inf')

    # Initial scope estimate: simple counting bound gives 105 for (7,5)
    # Start with a reasonable upper bound and try to reduce
    # The known best is 112, counting bound is 105
    # Let's start with scope_cap = 120 and try to find solutions, then reduce

    scope_cap = 120

    # Try multiple restarts with different random seeds
    attempt = 0
    while time_ok():
        attempt += 1

        # Decrease scope cap gradually as we find better solutions
        if best_score < float('inf'):
            scope_cap = min(scope_cap, best_score)
            # Also try slightly below best_score to see if we can improve
            if attempt % 5 == 0:
                scope_cap = best_score - 1

        rng = random.Random(attempt * 12345 + 67890)
        rows = solve_csp(rng, scope_cap)

        if rows is not None:
            if validate(rows):
                score = max(max(r) for r in rows)
                if score < best_score:
                    best_score = score
                    best_rows = [list(r) for r in rows]

                    # Try local search to improve further
                    if time_ok():
                        improved_rows, improved_score = local_search(
                            [list(r) for r in best_rows],
                            max_iterations=3000
                        )
                        if validate(improved_rows):
                            if improved_score < best_score:
                                best_score = improved_score
                                best_rows = improved_rows

                    # If we've reached the counting bound, we're done
                    if best_score <= 105:
                        break

        # If no solution found with current scope_cap, increase it slightly
        if rows is None and attempt % 3 == 0:
            scope_cap += 5
            # But don't go too high
            scope_cap = min(scope_cap, 150)

        # Occasionally do a local search on the best solution with different parameters
        if best_rows is not None and attempt % 10 == 0 and time_ok():
            improved_rows, improved_score = local_search(
                [list(r) for r in best_rows],
                max_iterations=2000
            )
            if validate(improved_rows):
                if improved_score < best_score:
                    best_score = improved_score
                    best_rows = improved_rows

    # Fallback: deterministic greedy if no CSP solution found
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
