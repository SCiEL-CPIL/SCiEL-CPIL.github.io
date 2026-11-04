"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    total_diffs = n_rows * k * (k + 1) // 2

    # Counting lower bound
    lower_bound = 0
    s = 0
    while s < total_diffs:
        lower_bound += 1
        s += lower_bound

    # Greedy baseline
    def greedy_construct(n_rows, k):
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

    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Simulated annealing with boolean array for O(1) diff lookup
    def simulated_annealing(rows, k, rng, max_iters=200000):
        rows = [list(r) for r in rows]
        n = len(rows)
        max_val = max(max(r) for r in rows)
        # Use a bytearray for fast diff tracking; size = max_val + 1
        # We'll dynamically resize if needed
        used = bytearray(max_val + 2)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used[d] = 1

        cur_max = max(max(r) for r in rows)
        best_max = cur_max
        best_rows = [list(r) for r in rows]

        T = 10.0  # initial temperature
        T_min = 0.01
        cooling = 0.9995

        for it in range(max_iters):
            if _time_left() < 3.0:
                break
            if it % 10000 == 0 and T < T_min:
                T = T_min

            # Choose a move type
            move_type = rng.random()
            if move_type < 0.4:
                # Shrink move: try to reduce the max element
                max_row_idx = rng.randrange(n)
                row = rows[max_row_idx]
                pos = len(row) - 1  # last element
                old_val = row[pos]
                # Try to reduce
                new_val = rng.randint(row[pos - 1] + 1, old_val - 1) if old_val > row[pos - 1] + 1 else -1
                if new_val < 0:
                    continue
                # Check if new diffs conflict
                old_diffs = [old_val - a for a in row[:pos]]
                new_diffs = [new_val - a for a in row[:pos]]
                # Temporarily remove old diffs
                for d in old_diffs:
                    used[d] = 0
                # Check new diffs
                conflict = False
                for d in new_diffs:
                    if d > len(used) - 1:
                        # Extend used
                        used.extend(bytearray(d - len(used) + 1))
                    if used[d]:
                        conflict = True
                        break
                if conflict:
                    # Restore old diffs
                    for d in old_diffs:
                        used[d] = 1
                    continue
                # Apply
                for d in new_diffs:
                    if d > len(used) - 1:
                        used.extend(bytearray(d - len(used) + 1))
                    used[d] = 1
                row[pos] = new_val
                new_max = max(max(r) for r in rows)
                delta = new_max - cur_max
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    cur_max = new_max
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = [list(r) for r in rows]
                else:
                    # Revert
                    row[pos] = old_val
                    for d in new_diffs:
                        used[d] = 0
                    for d in old_diffs:
                        used[d] = 1
            elif move_type < 0.7:
                # Perturb an interior element
                row_idx = rng.randrange(n)
                row = rows[row_idx]
                pos = rng.randint(1, len(row) - 2)  # interior
                old_val = row[pos]
                lo = row[pos - 1] + 1
                hi = row[pos + 1] - 1
                if hi <= lo:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue
                # Compute old and new diffs involving pos
                old_diffs = []
                new_diffs = []
                for j in range(len(row)):
                    if j == pos:
                        continue
                    if j < pos:
                        old_diffs.append(old_val - row[j])
                        new_diffs.append(new_val - row[j])
                    else:
                        old_diffs.append(row[j] - old_val)
                        new_diffs.append(row[j] - new_val)
                # Remove old diffs
                for d in old_diffs:
                    used[d] = 0
                # Check new diffs
                conflict = False
                for d in new_diffs:
                    if d > len(used) - 1:
                        used.extend(bytearray(d - len(used) + 1))
                    if used[d]:
                        conflict = True
                        break
                if conflict:
                    for d in old_diffs:
                        used[d] = 1
                    continue
                # Apply
                for d in new_diffs:
                    used[d] = 1
                row[pos] = new_val
                new_max = max(max(r) for r in rows)
                delta = new_max - cur_max
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    cur_max = new_max
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = [list(r) for r in rows]
                else:
                    # Revert
                    row[pos] = old_val
                    for d in new_diffs:
                        used[d] = 0
                    for d in old_diffs:
                        used[d] = 1
            else:
                # Swap two elements from different rows (if valid)
                r1, r2 = rng.sample(range(n), 2)
                p1 = rng.randint(1, len(rows[r1]) - 1)
                p2 = rng.randint(1, len(rows[r2]) - 1)
                v1 = rows[r1][p1]
                v2 = rows[r2][p2]
                # Check if swap maintains increasing order
                if p1 > 0 and v2 <= rows[r1][p1 - 1]:
                    continue
                if p1 < len(rows[r1]) - 1 and v2 >= rows[r1][p1 + 1]:
                    continue
                if p2 > 0 and v1 <= rows[r2][p2 - 1]:
                    continue
                if p2 < len(rows[r2]) - 1 and v1 >= rows[r2][p2 + 1]:
                    continue
                # Compute all affected diffs
                old_diffs = []
                new_diffs = []
                for j in range(len(rows[r1])):
                    if j == p1:
                        continue
                    if j < p1:
                        old_diffs.append(v1 - rows[r1][j])
                        new_diffs.append(v2 - rows[r1][j])
                    else:
                        old_diffs.append(rows[r1][j] - v1)
                        new_diffs.append(rows[r1][j] - v2)
                for j in range(len(rows[r2])):
                    if j == p2:
                        continue
                    if j < p2:
                        old_diffs.append(v2 - rows[r2][j])
                        new_diffs.append(v1 - rows[r2][j])
                    else:
                        old_diffs.append(rows[r2][j] - v2)
                        new_diffs.append(rows[r2][j] - v1)
                # Remove old
                for d in old_diffs:
                    used[d] = 0
                # Check new
                conflict = False
                for d in new_diffs:
                    if d > len(used) - 1:
                        used.extend(bytearray(d - len(used) + 1))
                    if used[d]:
                        conflict = True
                        break
                if conflict:
                    for d in old_diffs:
                        used[d] = 1
                    continue
                # Apply
                for d in new_diffs:
                    used[d] = 1
                rows[r1][p1] = v2
                rows[r2][p2] = v1
                new_max = max(max(r) for r in rows)
                delta = new_max - cur_max
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    cur_max = new_max
                    if cur_max < best_max:
                        best_max = cur_max
                        best_rows = [list(r) for r in rows]
                else:
                    # Revert
                    rows[r1][p1] = v1
                    rows[r2][p2] = v2
                    for d in new_diffs:
                        used[d] = 0
                    for d in old_diffs:
                        used[d] = 1

            T *= cooling

        return best_rows

    rng = random.Random(42)

    # Baseline
    baseline = greedy_construct(n_rows, k)
    best_rows = baseline
    best_max = max(max(r) for r in baseline)

    # Run simulated annealing from baseline
    if _time_left() > 5.0:
        improved = simulated_annealing(baseline, k, rng, max_iters=150000)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    # Try a few more restarts with different seeds if time permits
    for seed in [123, 456, 789]:
        if _time_left() < 10.0:
            break
        rng2 = random.Random(seed)
        # Start from a perturbed baseline
        perturbed = [list(r) for r in baseline]
        for _ in range(50):
            ri = rng2.randrange(n_rows)
            pi = rng2.randint(1, k)
            old = perturbed[ri][pi]
            lo = perturbed[ri][pi - 1] + 1
            hi = perturbed[ri][pi + 1] - 1 if pi < k else old + 10
            if hi > lo:
                new = rng2.randint(lo, min(hi, old + 5))
                if new != old:
                    perturbed[ri][pi] = new
        # Validate perturbed; if invalid, skip
        if not validate(perturbed, n_rows, k):
            continue
        improved = simulated_annealing(perturbed, k, rng2, max_iters=80000)
        if improved:
            new_max = max(max(r) for r in improved)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved

    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)

    return best_rows
