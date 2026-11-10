"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    deadline = time.time() + 115  # safety margin before 120s limit

    def diff_set(row):
        """Return the set of all positive differences within a row."""
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def used_from_rows(rows):
        """Return the union of all differences from all rows."""
        used = set()
        for r in rows:
            used |= diff_set(r)
        return used

    def validate(rows):
        """Check if all positive differences are distinct across all rows."""
        used = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    def greedy_row(used, temp=0.0, x_start=0):
        """Build one row greedily with optional temperature for exploration."""
        row = [0]
        x = x_start
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # With temperature, sometimes skip valid x to explore
                if temp > 0 and random.random() < temp and x < 200:
                    continue
                row.append(x)
                used |= new_diffs
        return row, used

    def find_replacement(rows, i, used_others, n_attempts=20):
        """
        Try to find a replacement for row i that is disjoint from used_others
        and has a smaller max value than the current row i.
        """
        current_max = max(rows[i])
        best_new_row = None
        best_new_max = current_max

        for _ in range(n_attempts):
            temp = random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
            # Try to build a row with a cap on the maximum value
            # We want to find a row with max < current_max if possible,
            # or at least one that fits.
            # Simple approach: just build greedily and check max.
            trial_row, _ = greedy_row(used_others, temp=temp)
            trial_max = max(trial_row)

            if trial_max < best_new_max:
                best_new_max = trial_max
                best_new_row = trial_row

            # Early exit if we found a very good one
            if trial_max < current_max - 5:
                break

        return best_new_row, best_new_max

    # Strategy 1: Pure greedy baseline
    used = set()
    rows_greedy = []
    for _ in range(n_rows):
        row, used = greedy_row(used, temp=0.0)
        rows_greedy.append(row)
    best_rows = list(rows_greedy)
    best_max = max_val(rows_greedy)

    # Strategy 2: Randomized greedy with varying temperatures
    random.seed(42)
    attempts = 0
    temps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8]

    while time.time() < deadline - 10 and attempts < 150:
        attempts += 1
        temp = temps[attempts % len(temps)]
        used = set()
        trial_rows = []
        success = True

        for _ in range(n_rows):
            if time.time() > deadline - 10:
                success = False
                break
            row, used = greedy_row(used, temp=temp)
            trial_rows.append(row)

        if not success:
            continue

        mv = max_val(trial_rows)
        if mv < best_max:
            best_max = mv
            best_rows = list(trial_rows)

    # Strategy 3: Local search - row replacement with disjointness check
    # This is the structural change: instead of tweaking integers, we replace entire rows.
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle row order to break symmetry in search
            indices = list(range(n_rows))
            random.shuffle(indices)

            for i in indices:
                if time.time() > deadline - 5:
                    break

                # Compute used set from all rows except row i
                other_rows = [r for j, r in enumerate(best_rows) if j != i]
                used_others = used_from_rows(other_rows)

                # Try to find a better row i
                new_row, new_max = find_replacement(best_rows, i, used_others)

                if new_row is not None:
                    # Verify that the new row is indeed disjoint from used_others
                    # (greedy_row ensures this, but let's be safe)
                    if not (diff_set(new_row) & used_others):
                        new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                        # Final validation
                        if validate(new_rows):
                            mv = max_val(new_rows)
                            if mv < best_max:
                                best_max = mv
                                best_rows = new_rows
                                improved = True

    # Strategy 4: "Compress" - try to reduce the max value of the best solution
    # by shifting the entire solution if possible? No, starts at 0.
    # Instead, try to "tighten" the rows by trying to reduce the last element.
    # This is harder. Let's stick to the row replacement.

    # One more pass with higher temperature to see if we can jump out
    if time.time() < deadline - 2:
        for temp in [0.8, 0.9, 1.0]:
            if time.time() > deadline - 2:
                break
            used = set()
            trial_rows = []
            success = True
            for _ in range(n_rows):
                row, used = greedy_row(used, temp=temp)
                trial_rows.append(row)
            if success:
                mv = max_val(trial_rows)
                if mv < best_max:
                    best_max = mv
                    best_rows = list(trial_rows)

    return best_rows
# EVOLVE-BLOCK-END
