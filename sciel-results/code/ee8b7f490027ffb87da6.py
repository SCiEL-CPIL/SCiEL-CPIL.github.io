import time
import random
import math
import sys

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    def _validate(rows):
        """Check if all differences are distinct across all rows."""
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def _scope(rows):
        """Return the largest entry across all rows."""
        return max(max(r) for r in rows)

    def _greedy_fallback():
        """Fast greedy construction to get a baseline."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
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
                        used.add(x - a)
            rows.append(row)
        return rows

    def _randomized_greedy_diff(bound=None):
        """Randomized greedy that builds rows by choosing differences."""
        if bound is None:
            bound = 200
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
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
                        used.add(x - a)
            rows.append(row)
        return rows

    def _get_diffs(row):
        """Get all differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def _reconstruct_row(diff_list, row_len):
        """
        Given a list of differences (of length row_len*(row_len-1)//2),
        try to reconstruct a valid row that starts at 0.
        Returns None if no valid reconstruction found.
        """
        n = row_len
        # The differences must be consistent with a set of n points
        # This is a hard constraint satisfaction problem
        # For now, return None - we'll use a different approach
        return None

    def _sa_optimize(initial_rows, time_limit):
        """
        Simulated annealing to optimize rows.
        We work in the space of row entries directly.
        """
        rows = [r[:] for r in initial_rows]
        n = len(rows)
        m = len(rows[0])

        # Build difference count map
        diff_count = {}
        for r in rows:
            for i in range(m):
                for j in range(i + 1, m):
                    d = r[j] - r[i]
                    if d in diff_count:
                        diff_count[d] += 1
                    else:
                        diff_count[d] = 1

        def get_collisions():
            c = 0
            for cnt in diff_count.values():
                if cnt > 1:
                    c += cnt - 1
            return c

        def get_scope():
            return max(max(r) for r in rows)

        current_collisions = get_collisions()
        current_scope = get_scope()
        best_collisions = current_collisions
        best_scope = current_scope
        best_rows = [r[:] for r in rows]

        T = 30.0
        cooling = 0.999
        start = time.time()

        while time.time() - start < time_limit:
            # Choose a random row and position (not the first)
            r_idx = random.randint(0, n - 1)
            i_idx = random.randint(1, m - 1)

            # Determine range for new value
            lower = rows[r_idx][i_idx - 1] + 1
            if i_idx < m - 1:
                upper = rows[r_idx][i_idx + 1] - 1
            else:
                upper = best_scope + 20

            if lower > upper:
                continue

            new_val = random.randint(lower, upper)
            old_val = rows[r_idx][i_idx]
            if new_val == old_val:
                continue

            # Remove old differences
            old_row = rows[r_idx]
            old_diffs = []
            for j in range(m):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    old_diffs.append(d)

            for d in old_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 1:
                    current_collisions -= 1
                if diff_count[d] == 0:
                    del diff_count[d]

            # Apply change
            rows[r_idx][i_idx] = new_val

            # Add new differences
            new_row = rows[r_idx]
            new_diffs = []
            for j in range(m):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    new_diffs.append(d)

            for d in new_diffs:
                if d in diff_count:
                    diff_count[d] += 1
                    if diff_count[d] == 2:
                        current_collisions += 1
                else:
                    diff_count[d] = 1

            new_scope = get_scope()

            # Acceptance criterion
            if current_collisions < best_collisions or \
               (current_collisions == best_collisions and new_scope < best_scope):
                best_collisions = current_collisions
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
                if best_collisions == 0:
                    pass  # Found valid solution

            elif current_collisions > best_collisions or \
                 (current_collisions == best_collisions and new_scope > best_scope):
                # Decide whether to accept based on temperature
                delta = (current_collisions - best_collisions) * 100 + (new_scope - best_scope)
                if delta < 0:
                    pass  # Accept
                else:
                    p = math.exp(-delta / max(T, 0.01))
                    if random.random() >= p:
                        # Reject - undo
                        rows[r_idx][i_idx] = old_val

                        # Remove new differences
                        for d in new_diffs:
                            diff_count[d] -= 1
                            if diff_count[d] == 1:
                                current_collisions -= 1
                            if diff_count[d] == 0:
                                del diff_count[d]

                        # Add old differences
                        for d in old_diffs:
                            if d in diff_count:
                                diff_count[d] += 1
                                if diff_count[d] == 2:
                                    current_collisions += 1
                            else:
                                diff_count[d] = 1
            else:
                # Same collisions, same scope - accept with some probability
                p = math.exp(-(new_scope - best_scope) / max(T, 0.01))
                if random.random() >= p:
                    # Reject - undo
                    rows[r_idx][i_idx] = old_val
                    for d in new_diffs:
                        diff_count[d] -= 1
                        if diff_count[d] == 1:
                            current_collisions -= 1
                        if diff_count[d] == 0:
                            del diff_count[d]
                    for d in old_diffs:
                        if d in diff_count:
                            diff_count[d] += 1
                            if diff_count[d] == 2:
                                current_collisions += 1
                        else:
                            diff_count[d] = 1

            T *= cooling
            if T < 0.5:
                T = 0.5

        return best_rows, best_scope, best_collisions

    def _row_reoptimize(rows, time_limit):
        """
        Try to optimize each row individually to reduce scope.
        """
        n = len(rows)
        m = len(rows[0])

        # Build global difference set
        used_diffs = set()
        for r in rows:
            for i in range(m):
                for j in range(i + 1, m):
                    used_diffs.add(r[j] - r[i])

        for r_idx in range(n):
            # Try to shrink this row
            for _ in range(100):
                if time.time() > time_limit:
                    break
                row = rows[r_idx][:]
                # Try to reduce the last element
                if len(row) < 2:
                    continue
                last = row[-1]
                # Try smaller values for the last element
                for new_last in range(last - 1, row[-2], -1):
                    # Check if this creates conflicts
                    valid = True
                    for i in range(len(row) - 1):
                        d = new_last - row[i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        # Accept
                        rows[r_idx][-1] = new_last
                        # Update used_diffs
                        for i in range(len(row) - 1):
                            used_diffs.discard(row[i] and (last - row[i]))
                            used_diffs.add(new_last - row[i])
                        break

        return rows

    # Main construction
    # Start with greedy baseline
    best_rows = _greedy_fallback()
    best_scope = _scope(best_rows)

    # Try multiple randomized starts and SA
    while time.time() < deadline:
        # Create initial state
        if random.random() < 0.4 and best_scope > 0:
            # Use best rows as starting point with small perturbations
            init_rows = [r[:] for r in best_rows]
            # Perturb a few entries
            for _ in range(random.randint(1, 3)):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                lower = init_rows[r_idx][i_idx - 1] + 1
                upper = init_rows[r_idx][i_idx + 1] - 1 if i_idx < row_len - 1 else best_scope + 10
                if lower <= upper:
                    init_rows[r_idx][i_idx] = random.randint(lower, upper)
        else:
            # Random initialization
            bound = best_scope + 10 if best_scope > 0 else 200
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)

        # Run SA
        sa_time = min(10.0, deadline - time.time())
        if sa_time < 2.0:
            break

        local_rows, local_scope, local_collisions = _sa_optimize(init_rows, sa_time)

        if local_collisions == 0 and local_scope < best_scope:
            best_rows = local_rows
            best_scope = local_scope
        elif local_collisions == 0 and best_scope > 0:
            # Even if not better, keep track
            pass

    # Final validation
    if not _validate(best_rows):
        # Fallback to greedy
        best_rows = _greedy_fallback()

    return best_rows
