"""Difference triangle set (7,5): minimize the largest mark (scope).

Strategy: Multi-restart greedy + SA with incremental diff tracking + scope squeeze.
"""
import time
import random
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Precompute: for a row of length k+1, the number of pairwise diffs
    num_diffs_per_row = k * (k + 1) // 2

    def row_diffs(row):
        """Return list of all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid_row(row):
        """Check strictly increasing and distinct internal differences."""
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                return False
        diffs = row_diffs(row)
        return len(diffs) == len(set(diffs))

    def calc_scope(rows):
        return max(max(row) for row in rows)

    def greedy_init(max_val=300):
        """Generate an initial valid set of rows using greedy with backtracking."""
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                for x in range(row[-1] + 1, max_val):
                    new_diffs = {x - a for a in row}
                    if new_diffs & used_diffs:
                        continue
                    test_row = row + [x]
                    if is_valid_row(test_row):
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    # Fallback with larger range
                    for x in range(row[-1] + 1, max_val + 200):
                        new_diffs = {x - a for a in row}
                        if new_diffs & used_diffs:
                            continue
                        test_row = row + [x]
                        if is_valid_row(test_row):
                            row.append(x)
                            used_diffs |= new_diffs
                            found = True
                            break
                if not found:
                    # Last resort: brute force search
                    for x in range(row[-1] + 1, 500):
                        new_diffs = {x - a for a in row}
                        if new_diffs & used_diffs:
                            continue
                        test_row = row + [x]
                        if is_valid_row(test_row):
                            row.append(x)
                            used_diffs |= new_diffs
                            found = True
                            break
            rows.append(row)
        return rows, used_diffs

    def sa_optimize(rows, used_diffs, time_limit, T_init=8.0, cooling=0.9995, min_T=0.05):
        """Run SA to minimize scope. Returns (best_rows, best_scope)."""
        current_rows = [r[:] for r in rows]
        current_used = set(used_diffs)
        best_rows = [r[:] for r in rows]
        best_scope = calc_scope(best_rows)
        current_scope = best_scope

        T = T_init
        steps = 0

        while time.time() < time_limit and T >= min_T:
            r_idx = random.randint(0, n_rows - 1)
            pos_idx = random.randint(1, k)

            current_val = current_rows[r_idx][pos_idx]

            # Choose new value
            if random.random() < 0.4 and current_val > 1:
                new_val = random.randint(1, current_val - 1)
            elif random.random() < 0.3:
                # Small perturbation
                delta = random.randint(-5, 5)
                new_val = current_val + delta
                if new_val <= 0:
                    new_val = 1
            else:
                new_val = random.randint(1, best_scope + 15)

            if new_val == current_val:
                continue

            # Check increasing constraint
            if pos_idx > 0 and new_val <= current_rows[r_idx][pos_idx - 1]:
                T *= cooling
                continue
            if pos_idx < k and new_val >= current_rows[r_idx][pos_idx + 1]:
                T *= cooling
                continue

            # Compute old and new diffs for this position
            old_diffs = set()
            new_diffs = set()
            for i in range(k + 1):
                if i != pos_idx:
                    d_old = abs(current_rows[r_idx][i] - current_val)
                    d_new = abs(current_rows[r_idx][i] - new_val)
                    old_diffs.add(d_old)
                    new_diffs.add(d_new)

            # Check internal distinctness of modified row
            test_row = current_rows[r_idx][:]
            test_row[pos_idx] = new_val
            if not is_valid_row(test_row):
                T *= cooling
                continue

            # Check global collisions
            remaining = current_used - old_diffs
            if new_diffs & remaining:
                T *= cooling
                continue

            # Valid move - apply
            old_val = current_val
            current_used -= old_diffs
            current_rows[r_idx][pos_idx] = new_val
            current_used |= new_diffs

            # Update scope
            new_scope = max(max(r) for r in current_rows)

            # Acceptance
            if new_scope <= best_scope or random.random() < pow(2.71828, -(new_scope - best_scope) / T):
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                # Revert
                current_used -= new_diffs
                current_rows[r_idx][pos_idx] = old_val
                current_used |= old_diffs

            T *= cooling
            steps += 1

        return best_rows, best_scope

    def scope_squeeze(rows, time_limit):
        """Try to reduce scope by forcing max element down."""
        current_rows = [r[:] for r in rows]
        current_scope = calc_scope(current_rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope

        # Build used_diffs
        used_diffs = set()
        for row in current_rows:
            used_diffs |= set(row_diffs(row))

        # Try to reduce the maximum element
        while time.time() < time_limit and best_scope > 0:
            # Find the row with the maximum element
            max_val = best_scope
            target_row = -1
            target_pos = -1
            for r in range(n_rows):
                if current_rows[r][k] == max_val:
                    target_row = r
                    target_pos = k
                    break

            if target_row < 0:
                break

            # Try to reduce
            improved = False
            for new_val in range(max_val - 1, 0, -1):
                if new_val <= current_rows[target_row][k - 1]:
                    break
                # Check validity
                old_diffs = set()
                new_diffs = set()
                for i in range(k + 1):
                    if i != target_pos:
                        old_diffs.add(abs(current_rows[target_row][i] - max_val))
                        new_diffs.add(abs(current_rows[target_row][i] - new_val))

                test_row = current_rows[target_row][:]
                test_row[target_pos] = new_val
                if not is_valid_row(test_row):
                    continue

                remaining = used_diffs - old_diffs
                if new_diffs & remaining:
                    continue

                # Apply
                used_diffs -= old_diffs
                current_rows[target_row][target_pos] = new_val
                used_diffs |= new_diffs
                current_scope = new_val
                improved = True
                break

            if not improved:
                # Try SA for a short burst to escape local optimum
                if time.time() < time_limit - 2:
                    sa_rows, sa_scope = sa_optimize(current_rows, used_diffs, time.time() + 3)
                    if sa_scope < best_scope:
                        best_scope = sa_scope
                        best_rows = sa_rows
                        current_rows = [r[:] for r in sa_rows]
                        used_diffs = set()
                        for row in current_rows:
                            used_diffs |= set(row_diffs(row))
                else:
                    break

            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]

        return best_rows, best_scope

    # Main: multiple restarts
    best_rows = None
    best_scope = float('inf')

    # Phase 1: Multiple greedy inits + SA
    num_restarts = 8
    restart_time = (deadline - start_time) * 0.5  # Half time for restarts

    for restart in range(num_restarts):
        if time.time() >= start_time + restart_time:
            break

        rows, used_diffs = greedy_init()
        scope = calc_scope(rows)

        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

        # SA on this init
        remaining = min(restart_time - (time.time() - start_time), 10.0)
        if remaining > 1:
            sa_rows, sa_scope = sa_optimize(rows, used_diffs, time.time() + remaining)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]

    # Phase 2: Scope squeeze on best
    if best_rows is not None and time.time() < deadline - 5:
        squeeze_rows, squeeze_scope = scope_squeeze(best_rows, deadline - 5)
        if squeeze_scope < best_scope:
            best_scope = squeeze_scope
            best_rows = squeeze_rows

    # Phase 3: Final SA polish
    if best_rows is not None and time.time() < deadline - 2:
        used_diffs = set()
        for row in best_rows:
            used_diffs |= set(row_diffs(row))
        final_rows, final_scope = sa_optimize(best_rows, used_diffs, deadline - 2, T_init=3.0, cooling=0.9998, min_T=0.1)
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = final_rows

    if best_rows is None:
        # Fallback
        best_rows, _ = greedy_init()

    return best_rows
