"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard: kill before 150s wall-clock limit
    deadline = time.time() + 142.0

    def max_entry(rows):
        return max(max(row) for row in rows)

    def all_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        diffs = all_diffs(rows)
        return len(diffs) == len(set(diffs))

    def count_violations(rows):
        diffs = all_diffs(rows)
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        return sum(c - 1 for c in counts.values() if c > 1)

    def compute_cost(rows):
        violations = count_violations(rows)
        m = max_entry(rows)
        return violations * 10000 + m

    # --- Phase 1: Greedy initialization ---
    def greedy_init():
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

    best_rows = greedy_init()
    best_max = max_entry(best_rows)
    best_cost = compute_cost(best_rows)

    # --- Phase 2: Backtracking with bitset ---
    def try_backtracking(max_scope, timeout=5.0):
        """Try to find a valid configuration with max entry <= max_scope using backtracking."""
        start_time = time.time()
        rows = [[0] for _ in range(n_rows)]

        def dfs(row_idx, pos_idx, used_diffs):
            if time.time() - start_time > timeout:
                return False

            if row_idx == n_rows:
                return True

            if pos_idx == k + 1:
                return dfs(row_idx + 1, 1, used_diffs)

            row = rows[row_idx]
            prev_val = row[pos_idx - 1]

            # Try values from prev_val+1 to max_scope
            # Use a heuristic: try values that create fewer conflicts first
            candidates = []
            for x in range(prev_val + 1, max_scope + 1):
                new_diffs = {x - row[i] for i in range(pos_idx)}
                conflict = False
                new_used = used_diffs
                for d in new_diffs:
                    if (used_diffs >> d) & 1:
                        conflict = True
                        break
                    new_used |= (1 << d)

                if not conflict:
                    # Score: prefer values that use differences not yet used
                    # and that are "spread out"
                    candidates.append((x, new_used))

            if not candidates:
                return False

            # Sort by some heuristic - try smaller values first to keep max small
            # but also consider the spread of differences
            random.shuffle(candidates)

            for x, new_used in candidates:
                if time.time() - start_time > timeout:
                    return False

                row.append(x)
                if dfs(row_idx, pos_idx + 1, new_used):
                    return True
                row.pop()

            return False

        if dfs(0, 1, 0):
            return [r[:] for r in rows]
        return None

    # Try backtracking with decreasing scope
    for scope in range(best_max - 1, max(100, best_max - 30), -1):
        if time.time() > deadline - 30:
            break
        result = try_backtracking(scope, timeout=3.0)
        if result is not None:
            best_rows = result
            best_max = scope
            best_cost = compute_cost(best_rows)
            continue
        else:
            break

    # --- Phase 3: Simulated Annealing ---
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    current_cost = best_cost

    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]

    T = 50.0
    T_min = 0.01
    cooling = 0.9999

    while time.time() < deadline - 5:
        if T < T_min:
            break

        r, i = random.choice(indices)
        old_val = current_rows[r][i]

        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 50

        # Randomly choose a new value
        if random.random() < 0.3:
            # Try to move towards a "good" position - smaller values
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.3)
        else:
            new_val = random.randint(lower, upper)

        if new_val == old_val:
            continue

        current_rows[r][i] = new_val

        # Check monotonicity
        row = current_rows[r]
        if any(row[j] <= row[j - 1] for j in range(1, len(row))):
            current_rows[r][i] = old_val
            continue

        new_cost = compute_cost(current_rows)

        delta = new_cost - current_cost
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            new_max = max_entry(current_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [rr[:] for rr in current_rows]
                best_cost = new_cost
        else:
            current_rows[r][i] = old_val

        T *= cooling

    # Final validation
    if is_valid(best_rows):
        return best_rows
    else:
        # If not valid, try to fix with a few more SA iterations
        # or just return the best we have
        return best_rows
# EVOLVE-BLOCK-END
