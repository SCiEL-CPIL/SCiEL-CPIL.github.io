"""Difference triangle set (n_rows, k): minimize the largest entry (scope)."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    TIME_MARGIN = 5.0  # safety margin before the 120s kill

    # Parameters for the specific task: 7 rows of 6 increasing integers starting at 0
    # So each row has k+1 = 7 entries: a_0=0 < a_1 < ... < a_6
    # k = 6 means 6 differences from 0, plus all pairwise differences

    def elapsed():
        return time.time() - start_time

    def time_left():
        return DEADLINE - TIME_MARGIN - elapsed()

    def all_diffs(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d > 0:
                    diffs.add(d)
        return diffs

    def is_valid(rows):
        """Check if all positive differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for d in all_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        """Return the largest entry across all rows."""
        return max(max(row) for row in rows)

    def greedy_row(used, max_val, start_from=1):
        """
        Greedily construct a row of k+1 entries starting at 0,
        using the smallest possible next mark whose differences are unused.
        Returns None if no valid row can be constructed up to max_val.
        """
        row = [0]
        x = start_from
        while len(row) < k + 1:
            if x > max_val:
                return None
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                x += 1
                continue
            row.append(x)
            used = used | new_diffs
            x += 1
        return row

    def random_row(used, max_val):
        """
        Try to construct a random-ish row by sampling candidate positions.
        Returns None on failure.
        """
        row = [0]
        current_max = 0
        attempts = 0
        max_attempts = 200
        while len(row) < k + 1 and attempts < max_attempts:
            attempts += 1
            # Choose a random candidate in a range that's likely to work
            lo = current_max + 1
            hi = min(max_val, current_max + 30)
            if hi < lo:
                return None
            x = random.randint(lo, hi)
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            row.append(x)
            used = used | new_diffs
            current_max = x
        if len(row) < k + 1:
            return None
        return row

    def local_search(rows, used, max_iter=500):
        """
        Try to reduce the scope by moving individual entries downward.
        Returns (new_rows, new_used, new_scope) or None if no improvement.
        """
        rows = [list(r) for r in rows]
        used = set(used)
        best_scope = scope(rows)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time_left() > 1.0:
            improved = False
            iter_count += 1
            for r_idx in range(n_rows):
                for pos in range(1, len(rows[r_idx])):  # skip position 0
                    old_val = rows[r_idx][pos]
                    # Try to reduce this entry
                    for new_val in range(old_val - 1, 0, -1):
                        if new_val <= rows[r_idx][pos - 1]:
                            break
                        # Check if new_val is greater than previous entry
                        # and maintains strict increase
                        if new_val <= rows[r_idx][pos - 1]:
                            break
                        # Check all subsequent entries are still > new_val
                        if pos + 1 < len(rows[r_idx]) and new_val >= rows[r_idx][pos + 1]:
                            continue
                        # Compute new diffs for this row
                        new_row = rows[r_idx][:]
                        new_row[pos] = new_val
                        new_diffs = all_diffs(new_row)
                        # Remove old diffs for this row from used
                        old_diffs = all_diffs(rows[r_idx])
                        used_temp = used - old_diffs
                        if new_diffs & used_temp:
                            continue
                        # Accept
                        used = used_temp | new_diffs
                        rows[r_idx] = new_row
                        improved = True
                        break
        new_scope = scope(rows)
        return rows, used, new_scope

    def simulated_annealing(rows, used, start_temp=10.0, end_temp=0.1, max_iter=2000):
        """
        Simulated annealing with row-replacement and entry-shrinking moves.
        Returns the best (rows, used, scope) found.
        """
        rows = [list(r) for r in rows]
        used = set(used)
        best_rows = [list(r) for r in rows]
        best_used = set(used)
        best_scope = scope(rows)
        current_scope = best_scope

        for it in range(max_iter):
            if time_left() < 1.0:
                break
            temp = start_temp * (end_temp / start_temp) ** (it / max_iter)

            # Choose a move type
            move_type = random.random()
            new_rows = [list(r) for r in rows]
            new_used = set(used)

            if move_type < 0.5:
                # Row replacement: rebuild one row randomly
                r_idx = random.randint(0, n_rows - 1)
                old_diffs = all_diffs(rows[r_idx])
                new_used = used - old_diffs
                new_row = random_row(new_used, current_scope + 20)
                if new_row is None:
                    continue
                new_rows[r_idx] = new_row
                new_used = new_used | all_diffs(new_row)
            elif move_type < 0.8:
                # Entry shift: move one entry in one row
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, len(rows[r_idx]) - 1)
                old_val = rows[r_idx][pos]
                delta = random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
                if new_val <= rows[r_idx][pos - 1]:
                    continue
                if pos + 1 < len(rows[r_idx]) and new_val >= rows[r_idx][pos + 1]:
                    continue
                new_rows[r_idx] = list(rows[r_idx])
                new_rows[r_idx][pos] = new_val
                old_diffs = all_diffs(rows[r_idx])
                new_diffs = all_diffs(new_rows[r_idx])
                new_used = used - old_diffs
                if new_diffs & new_used:
                    continue
                new_used = new_used | new_diffs
            else:
                # Local shrink: try to reduce a random entry
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, len(rows[r_idx]) - 1)
                old_val = rows[r_idx][pos]
                if old_val <= rows[r_idx][pos - 1] + 1:
                    continue
                new_val = random.randint(rows[r_idx][pos - 1] + 1, old_val - 1)
                if pos + 1 < len(rows[r_idx]) and new_val >= rows[r_idx][pos + 1]:
                    continue
                new_rows[r_idx] = list(rows[r_idx])
                new_rows[r_idx][pos] = new_val
                old_diffs = all_diffs(rows[r_idx])
                new_diffs = all_diffs(new_rows[r_idx])
                new_used = used - old_diffs
                if new_diffs & new_used:
                    continue
                new_used = new_used | new_diffs

            new_scope = scope(new_rows)
            delta_scope = new_scope - current_scope

            if delta_scope <= 0:
                # Accept improvement
                rows = new_rows
                used = new_used
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                    best_used = set(used)
            else:
                # Accept with probability exp(-delta/temp)
                import math
                if temp > 0:
                    prob = math.exp(-delta_scope / max(temp, 0.01))
                    if random.random() < prob:
                        rows = new_rows
                        used = new_used
                        current_scope = new_scope

        return best_rows, best_used, best_scope

    # Main algorithm
    random.seed(42)

    # Step 1: Initial greedy construction
    best_rows = None
    best_scope = float('inf')

    for restart in range(20):
        if time_left() < 5.0:
            break

        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = greedy_row(used, 200)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= all_diffs(row)

        if not success:
            continue

        cur_scope = scope(rows)
        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [list(r) for r in rows]

        # Step 2: Local search to shrink
        if time_left() < 5.0:
            break
        ls_rows, ls_used, ls_scope = local_search(rows, used, max_iter=300)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows

        # Step 3: Simulated annealing
        if time_left() < 10.0:
            break
        sa_rows, sa_used, sa_scope = simulated_annealing(
            best_rows, set(used) if best_rows == rows else all_diffs_union(best_rows),
            start_temp=8.0, end_temp=0.05, max_iter=1500
        )
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    # Helper to get union of all diffs
    def all_diffs_union(rows):
        s = set()
        for row in rows:
            s |= all_diffs(row)
        return s

    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, 500)
            rows.append(row)
            used |= all_diffs(row)
        return rows

    return best_rows
# EVOLVE-BLOCK-END
