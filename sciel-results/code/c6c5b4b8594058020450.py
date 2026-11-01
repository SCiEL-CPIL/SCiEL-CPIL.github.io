import time
import random
import copy
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    def time_left():
        return deadline - time.time()

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
    best_scope = max(max(row) for row in best_rows)

    def calculate_initial_state(rows):
        diff_counter = Counter()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff_counter[row[j] - row[i]] += 1
        conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
        scope = max(max(row) for row in rows)
        return diff_counter, conflicts, scope

    def get_scope(rows):
        return max(max(row) for row in rows)

    def try_shrink_element(current_rows, diff_counter, r, j):
        """Try to reduce element (r,j) by 1. Returns True if successful."""
        old_val = current_rows[r][j]
        new_val = old_val - 1

        if new_val <= current_rows[r][j - 1]:
            return False

        # Remove old differences
        old_diffs = []
        for i in range(j):
            d = old_val - current_rows[r][i]
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        for i in range(j + 1, len(current_rows[r])):
            d = current_rows[r][i] - old_val
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]

        # Apply new value
        current_rows[r][j] = new_val

        # Add new differences
        for i in range(j):
            d = new_val - current_rows[r][i]
            diff_counter[d] += 1
        for i in range(j + 1, len(current_rows[r])):
            d = current_rows[r][i] - new_val
            diff_counter[d] += 1

        new_conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
        if new_conflicts == 0:
            return True
        else:
            # Revert
            current_rows[r][j] = old_val
            for d in old_diffs:
                diff_counter[d] += 1
            for i in range(j):
                d = new_val - current_rows[r][i]
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for i in range(j + 1, len(current_rows[r])):
                d = current_rows[r][i] - new_val
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            return False

    def systematic_shrink(current_rows, diff_counter, max_iterations=10000):
        """Systematically try to shrink elements from highest to lowest."""
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time_left() > 5.0:
            improved = False
            iterations += 1
            # Try each row, from last position to first
            for r in range(n_rows):
                for j in range(len(current_rows[r]) - 1, 0, -1):
                    if try_shrink_element(current_rows, diff_counter, r, j):
                        improved = True
                        new_scope = get_scope(current_rows)
                        if new_scope < best_scope:
                            nonlocal best_scope, best_rows
                            best_scope = new_scope
                            best_rows = copy.deepcopy(current_rows)
        return current_rows

    def sa_optimize(current_rows, diff_counter, curr_conflicts, curr_scope, time_budget):
        """Simulated annealing with fast conflict tracking."""
        T = max(5.0, curr_scope * 0.1)
        T_min = 0.5
        cooling = 0.997
        start = time.time()

        while T > T_min and time.time() - start < time_budget and time_left() > 5.0:
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k)

            old_val = current_rows[r][j]
            min_val = current_rows[r][j - 1] + 1
            max_val = max(old_val + 15, best_scope)
            if j < k:
                max_val = min(max_val, current_rows[r][j + 1] - 1)

            if min_val > max_val:
                T *= cooling
                continue

            new_val = random.randint(min_val, max_val)
            if new_val == old_val:
                T *= cooling
                continue

            # Remove old diffs
            old_diffs = []
            for i in range(j):
                d = old_val - current_rows[r][i]
                old_diffs.append(d)
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for i in range(j + 1, len(current_rows[r])):
                d = current_rows[r][i] - old_val
                old_diffs.append(d)
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]

            # Apply new
            current_rows[r][j] = new_val

            # Add new diffs
            for i in range(j):
                d = new_val - current_rows[r][i]
                diff_counter[d] += 1
            for i in range(j + 1, len(current_rows[r])):
                d = current_rows[r][i] - new_val
                diff_counter[d] += 1

            new_scope = get_scope(current_rows)
            new_conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)

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
                curr_scope = new_scope
                curr_conflicts = new_conflicts
                if new_conflicts == 0 and new_scope < best_scope:
                    nonlocal best_scope, best_rows
                    best_scope = new_scope
                    best_rows = copy.deepcopy(current_rows)
            else:
                # Revert
                current_rows[r][j] = old_val
                for d in old_diffs:
                    diff_counter[d] += 1
                for i in range(j):
                    d = new_val - current_rows[r][i]
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for i in range(j + 1, len(current_rows[r])):
                    d = current_rows[r][i] - new_val
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]

            T *= cooling
            if T < T_min:
                T = T_min

        return current_rows, diff_counter, curr_conflicts, curr_scope

    # Main search loop with iterated local search
    while time_left() > 10.0:
        # Phase 1: Systematic shrink
        current_rows = copy.deepcopy(best_rows)
        diff_counter, curr_conflicts, curr_scope = calculate_initial_state(current_rows)
        current_rows = systematic_shrink(current_rows, diff_counter)

        # Phase 2: SA optimization
        sa_time = min(30.0, time_left() * 0.4)
        current_rows, diff_counter, curr_conflicts, curr_scope = sa_optimize(
            current_rows, diff_counter, curr_conflicts, curr_scope, sa_time
        )

        # Phase 3: Another systematic shrink after SA
        if time_left() > 10.0:
            current_rows = systematic_shrink(current_rows, diff_counter)

        # Check if we improved
        new_scope = get_scope(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = copy.deepcopy(current_rows)

        # If we haven't improved in a while, do a larger perturbation
        # (implicit in the restart above since we always start from best)

    return best_rows
