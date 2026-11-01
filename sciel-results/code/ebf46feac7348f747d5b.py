"""Combined approach: targeted descent + simulated annealing for difference triangle sets."""
import time
import random
import math
import copy
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin under 150s kill

    # Helper functions
    def row_diffs(row):
        """All positive differences a[j]-a[i] for i<j in one row."""
        d = []
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                d.append(row[j] - ai)
        return d

    def build_counter(rows):
        """Build a Counter of all differences across all rows."""
        c = Counter()
        for row in rows:
            for d in row_diffs(row):
                c[d] += 1
        return c

    def conflicts_from_counter(c):
        """Count total conflicts from a difference counter."""
        return sum(v - 1 for v in c.values() if v > 1)

    def scope(rows):
        return max(r[-1] for r in rows)

    def is_valid(rows):
        c = build_counter(rows)
        return all(v == 1 for v in c.values())

    # Phase 1: Greedy construction
    def greedy_construct():
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

    best_rows = greedy_construct()
    best_scope = scope(best_rows)
    current_rows = copy.deepcopy(best_rows)

    # Initialize state
    diff_counter = build_counter(current_rows)
    curr_conflicts = conflicts_from_counter(diff_counter)
    curr_scope = scope(current_rows)

    # Targeted descent: try to reduce the largest elements
    def targeted_descent(rows, diff_counter, max_iterations=500):
        """
        Directly minimize the global maximum by attempting to reduce
        the largest elements in the rows containing them.
        """
        rows = copy.deepcopy(rows)
        diff_counter = Counter(diff_counter)
        improved = False

        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            # Find the row and position with the largest value
            max_val = -1
            max_row = -1
            max_pos = -1
            for i in range(len(rows)):
                for j in range(len(rows[i])):
                    if rows[i][j] > max_val:
                        max_val = rows[i][j]
                        max_row = i
                        max_pos = j

            if max_row == -1:
                break

            # Try to reduce this value
            old_val = rows[max_row][max_pos]
            reduced = False

            # Try values from old_val - 1 down to rows[max_row][max_pos-1] + 1
            for new_val in range(old_val - 1, max(rows[max_row][max_pos - 1] + 1, 0), -1):
                # Remove old differences involving (max_row, max_pos)
                old_diffs = []
                for i in range(max_pos):
                    d = old_val - rows[max_row][i]
                    old_diffs.append(d)
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for i in range(max_pos + 1, len(rows[max_row])):
                    d = rows[max_row][i] - old_val
                    old_diffs.append(d)
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]

                # Apply new value
                rows[max_row][max_pos] = new_val

                # Add new differences
                for i in range(max_pos):
                    d = new_val - rows[max_row][i]
                    diff_counter[d] += 1
                for i in range(max_pos + 1, len(rows[max_row])):
                    d = rows[max_row][i] - new_val
                    diff_counter[d] += 1

                # Check if valid
                new_conflicts = conflicts_from_counter(diff_counter)
                if new_conflicts == 0:
                    reduced = True
                    break
                else:
                    # Revert
                    rows[max_row][max_pos] = old_val
                    for d in old_diffs:
                        diff_counter[d] += 1
                    for i in range(max_pos):
                        d = new_val - rows[max_row][i]
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]
                    for i in range(max_pos + 1, len(rows[max_row])):
                        d = rows[max_row][i] - new_val
                        diff_counter[d] -= 1
                        if diff_counter[d] == 0:
                            del diff_counter[d]

            if not reduced:
                break

            improved = True

        return rows, diff_counter, improved

    # Simulated annealing step
    def sa_step(rows, diff_counter, T):
        """
        Perform one simulated annealing step: perturb a random element
        and accept based on energy change.
        """
        rows = copy.deepcopy(rows)
        diff_counter = Counter(diff_counter)

        r = random.randint(0, n_rows - 1)
        j = random.randint(1, k)

        old_val = rows[r][j]
        min_val = rows[r][j - 1] + 1
        max_val = max(old_val + 5, best_scope)
        if j < k:
            max_val = min(max_val, rows[r][j + 1] - 1)

        if min_val > max_val:
            return rows, diff_counter, False

        new_val = random.randint(min_val, max_val)
        if new_val == old_val:
            return rows, diff_counter, False

        # Remove old diffs
        old_diffs = []
        for i in range(j):
            d = old_val - rows[r][i]
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        for i in range(j + 1, len(rows[r])):
            d = rows[r][i] - old_val
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]

        # Apply new
        rows[r][j] = new_val

        # Add new diffs
        for i in range(j):
            d = new_val - rows[r][i]
            diff_counter[d] += 1
        for i in range(j + 1, len(rows[r])):
            d = rows[r][i] - new_val
            diff_counter[d] += 1

        new_scope = scope(rows)
        new_conflicts = conflicts_from_counter(diff_counter)

        delta_scope = new_scope - curr_scope
        delta_conflicts = new_conflicts - curr_conflicts
        delta_energy = delta_scope + 10 * delta_conflicts

        accept = False
        if delta_energy < 0:
            accept = True
        else:
            if T > 0:
                accept = random.random() < math.exp(-delta_energy / T)

        if accept:
            return rows, diff_counter, True
        else:
            # Revert
            rows[r][j] = old_val
            for d in old_diffs:
                diff_counter[d] += 1
            for i in range(j):
                d = new_val - rows[r][i]
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for i in range(j + 1, len(rows[r])):
                d = rows[r][i] - new_val
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            return rows, diff_counter, False

    # Main optimization loop
    T = 30.0
    cooling = 0.995

    while time.time() < deadline:
        # Try targeted descent first
        if time.time() < deadline - 5:
            new_rows, new_counter, improved = targeted_descent(current_rows, diff_counter, max_iterations=100)
            if improved:
                current_rows = new_rows
                diff_counter = new_counter
                curr_scope = scope(current_rows)
                curr_conflicts = conflicts_from_counter(diff_counter)
                if curr_conflicts == 0 and curr_scope < best_scope:
                    best_scope = curr_scope
                    best_rows = copy.deepcopy(current_rows)
                    T = max(T, 15.0)
                    continue

        # Then do SA steps
        if time.time() > deadline - 2:
            break

        for _ in range(50):
            if time.time() > deadline - 2:
                break

            new_rows, new_counter, accepted = sa_step(current_rows, diff_counter, T)
            if accepted:
                current_rows = new_rows
                diff_counter = new_counter
                curr_scope = scope(current_rows)
                curr_conflicts = conflicts_from_counter(diff_counter)

                if curr_conflicts == 0 and curr_scope < best_scope:
                    best_scope = curr_scope
                    best_rows = copy.deepcopy(current_rows)

        T *= cooling
        if T < 0.5:
            T = 0.5

    return best_rows
