import time
import random
import math
import sys

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    def greedy_construct(seed=None, ambition=1):
        """Greedy construction with optional randomization."""
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for r_idx in range(n_rows):
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
                elif x > 200:
                    # Force: try skipping
                    x += 1
            rows.append(row)
        return rows

    def get_diffs(rows):
        """Get all differences and check validity."""
        diff_count = {}
        collisions = 0
        max_val = 0
        for r in rows:
            if r[-1] > max_val:
                max_val = r[-1]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in diff_count:
                        diff_count[d] += 1
                        if diff_count[d] == 2:
                            collisions += 1
                    else:
                        diff_count[d] = 1
        return diff_count, collisions, max_val

    def is_valid(rows):
        _, col, _ = get_diffs(rows)
        return col == 0

    # --- Fast SA state with O(1) max updates ---
    class SAState:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.row_max = [0] * self.n_rows
            self._rebuild()

        def _rebuild(self):
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.row_max = [0] * self.n_rows
            for idx, r in enumerate(self.rows):
                self.row_max[idx] = r[-1]
                if r[-1] > self.max_val:
                    self.max_val = r[-1]
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        if d in self.diff_count:
                            self.diff_count[d] += 1
                            if self.diff_count[d] == 2:
                                self.collisions += 1
                        else:
                            self.diff_count[d] = 1

        def try_move(self, r_idx, i_idx, new_val):
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return None

            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx - 1]:
                return None
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx + 1]:
                return None

            old_row = self.rows[r_idx]
            delta_collisions = 0

            # Collect changes
            changes = {}
            for j in range(self.row_len):
                if j != i_idx:
                    d_old = abs(old_val - old_row[j])
                    d_new = abs(new_val - old_row[j])
                    if d_old == d_new:
                        continue
                    if d_old not in changes:
                        changes[d_old] = 0
                    changes[d_old] -= 1
                    if d_new not in changes:
                        changes[d_new] = 0
                    changes[d_new] += 1

            for d, delta in changes.items():
                current_count = self.diff_count.get(d, 0)
                new_count = current_count + delta
                if new_count < 0:
                    return None
                old_contrib = max(0, current_count - 1)
                new_contrib = max(0, new_count - 1)
                delta_collisions += (new_contrib - old_contrib)

            # Apply
            self.rows[r_idx][i_idx] = new_val
            for d, delta in changes.items():
                if delta == 0:
                    continue
                if delta < 0:
                    if d in self.diff_count:
                        self.diff_count[d] += delta
                        if self.diff_count[d] == 0:
                            del self.diff_count[d]
                else:
                    if d in self.diff_count:
                        self.diff_count[d] += delta
                    else:
                        self.diff_count[d] = delta

            self.collisions += delta_collisions

            # Update max
            old_max = self.max_val
            old_row_max = self.row_max[r_idx]
            new_row_max = self.rows[r_idx][-1] if i_idx != self.row_len - 1 else new_val
            # Actually row max is always the last element since strictly increasing
            new_row_max = self.rows[r_idx][-1]
            if i_idx == self.row_len - 1:
                self.row_max[r_idx] = new_val
            # row_max only changes if i_idx is last
            if i_idx == self.row_len - 1:
                if new_val > self.max_val:
                    self.max_val = new_val
                elif old_val == self.max_val:
                    # Find new max
                    self.max_val = max(self.row_max)

            return old_val, delta_collisions, (self.max_val - old_max)

        def undo_move(self, r_idx, i_idx, old_val, delta_collisions, delta_max):
            new_val = self.rows[r_idx][i_idx]
            old_row = self.rows[r_idx]

            changes = {}
            for j in range(self.row_len):
                if j != i_idx:
                    d_new = abs(new_val - old_row[j])
                    d_old = abs(old_val - old_row[j])
                    if d_new == d_old:
                        continue
                    if d_new not in changes:
                        changes[d_new] = 0
                    changes[d_new] -= 1
                    if d_old not in changes:
                        changes[d_old] = 0
                    changes[d_old] += 1

            for d, delta in changes.items():
                if delta == 0:
                    continue
                if delta < 0:
                    if d in self.diff_count:
                        self.diff_count[d] += delta
                        if self.diff_count[d] == 0:
                            del self.diff_count[d]
                else:
                    if d in self.diff_count:
                        self.diff_count[d] += delta
                    else:
                        self.diff_count[d] = delta

            self.rows[r_idx][i_idx] = old_val
            self.collisions -= delta_collisions

            if i_idx == self.row_len - 1:
                self.row_max[r_idx] = old_val
                self.max_val = max(self.row_max)

    def sa_optimize(initial_rows, time_budget):
        """Simulated annealing to optimize."""
        state = SAState(initial_rows)
        cur_col, cur_max = state.collisions, state.max_val
        best_rows = [r[:] for r in state.rows]
        best_col, best_max = cur_col, cur_max

        T = 200.0
        cooling = 0.9995
        end_time = time.time() + time_budget
        last_improve = time.time()

        while time.time() < end_time:
            # Check time
            if time.time() > deadline - 2:
                break

            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)

            lower = state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = state.rows[r_idx][i_idx + 1] - 1
            else:
                upper = cur_max + 20

            if lower > upper:
                continue

            # Bias toward reducing max: prefer smaller values
            if cur_col == 0:
                # In valid region, bias toward compression
                bias_center = (lower + upper) // 2
                new_val = random.randint(lower, min(upper, bias_center + (upper - lower) // 4))
            else:
                new_val = random.randint(lower, upper)

            if new_val == state.rows[r_idx][i_idx]:
                continue

            res = state.try_move(r_idx, i_idx, new_val)
            if res is None:
                continue

            old_val, d_col, d_max = res
            new_col = cur_col + d_col
            new_max = cur_max + d_max

            accepted = False
            if new_col < cur_col:
                accepted = True
            elif new_col == cur_col:
                if new_max < cur_max:
                    accepted = True
                elif new_max == cur_max:
                    accepted = random.random() < 0.3
                else:
                    if cur_col == 0:
                        delta = new_max - cur_max
                        if delta <= 15:
                            accepted = random.random() < math.exp(-delta / max(T, 1.0))
                    else:
                        delta = new_max - cur_max
                        if delta <= 40:
                            accepted = random.random() < math.exp(-delta / max(T * 3, 1.0))
            else:
                delta_col = new_col - cur_col
                penalty = delta_col * 80
                if penalty < 800:
                    accepted = random.random() < math.exp(-penalty / max(T, 1.0))

            if accepted:
                cur_col = new_col
                cur_max = new_max
                if (cur_col, cur_max) < (best_col, best_max):
                    best_col = cur_col
                    best_max = cur_max
                    best_rows = [r[:] for r in state.rows]
                    last_improve = time.time()
            else:
                state.undo_move(r_idx, i_idx, old_val, d_col, d_max)

            T *= cooling
            if T < 0.5:
                T = 0.5

            # Stagnation restart
            if time.time() - last_improve > 8.0 and cur_col > 0:
                # Perturb
                for _ in range(n_rows * 3):
                    rr = random.randint(0, n_rows - 1)
                    ii = random.randint(1, row_len - 1)
                    lo = state.rows[rr][ii - 1] + 1
                    hi = state.rows[rr][ii + 1] - 1 if ii < row_len - 1 else cur_max + 10
                    if lo < hi:
                        nv = random.randint(lo, hi)
                        r = state.try_move(rr, ii, nv)
                        if r:
                            ov, dc, dm = r
                            if dc > 0:
                                state.undo_move(rr, ii, ov, dc, dm)
                last_improve = time.time()
                T = max(T, 80.0)

        return best_rows, best_col, best_max

    def compress_max(rows, time_budget):
        """Try to reduce the maximum entry in a valid solution."""
        state = SAState(rows)
        best_rows = [r[:] for r in state.rows]
        best_max = state.max_val
        end_time = time.time() + time_budget

        while time.time() < end_time and time.time() < deadline - 1:
            # Find the row with the largest last element
            max_row_idx = max(range(n_rows), key=lambda i: state.rows[i][-1])
            last_idx = row_len - 1
            current_last = state.rows[max_row_idx][last_idx]

            # Try to reduce the last element
            lower = state.rows[max_row_idx][last_idx - 1] + 1
            improved = False

            for new_last in range(lower, current_last):
                if time.time() > end_time:
                    break
                res = state.try_move(max_row_idx, last_idx, new_last)
                if res is None:
                    continue
                old_val, d_col, d_max = res
                if d_col == 0 and d_max < 0:
                    # Accept: reduces max without creating collisions
                    best_max = state.max_val
                    best_rows = [r[:] for r in state.rows]
                    improved = True
                    break
                else:
                    state.undo_move(max_row_idx, last_idx, old_val, d_col, d_max)

            if not improved:
                # Try other rows or perturb
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                lower = state.rows[r_idx][i_idx - 1] + 1
                upper = state.rows[r_idx][i_idx + 1] - 1 if i_idx < row_len - 1 else state.rows[r_idx][-1] - 1
                if lower < upper:
                    new_val = random.randint(lower, upper)
                    res = state.try_move(r_idx, i_idx, new_val)
                    if res:
                        old_val, d_col, d_max = res
                        if d_col == 0:
                            # Keep if it doesn't increase max
                            if state.max_val <= best_max:
                                pass  # accept
                            else:
                                state.undo_move(r_idx, i_idx, old_val, d_col, d_max)
                        else:
                            state.undo_move(r_idx, i_idx, old_val, d_col, d_max)

        return best_rows, best_max

    # --- Main search ---
    best_rows = None
    best_max = float('inf')

    # Phase 1: Multiple greedy starts + SA
    n_starts = 3
    per_start_time = 25.0

    for s in range(n_starts):
        if time.time() > deadline - 5:
            break
        # Different greedy strategies
        if s == 0:
            init_rows = greedy_construct(seed=42)
        elif s == 1:
            init_rows = greedy_construct(seed=random.randint(1, 10000))
        else:
            init_rows = greedy_construct(seed=random.randint(10001, 20000))

        col, mx = get_diffs(init_rows)[1], get_diffs(init_rows)[2]

        # SA to fix collisions and optimize
        sa_rows, sa_col, sa_max = sa_optimize(init_rows, per_start_time)

        if sa_col == 0 and sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows[:]
        elif best_rows is None and sa_col == 0:
            best_rows = sa_rows[:]
            best_max = sa_max

    # Phase 2: Compress the best valid solution
    if best_rows is not None and time.time() < deadline - 10:
        compress_budget = min(40.0, deadline - 10 - time.time())
        if compress_budget > 5:
            best_rows, best_max = compress_max(best_rows, compress_budget)

    # Phase 3: Final SA polish on the compressed solution
    if best_rows is not None and time.time() < deadline - 5:
        polish_budget = min(20.0, deadline - 5 - time.time())
        if polish_budget > 3:
            polished, pcol, pmax = sa_optimize(best_rows, polish_budget)
            if pcol == 0 and pmax < best_max:
                best_rows = polished
                best_max = pmax

    # Fallback: if we somehow don't have a valid solution, use greedy
    if best_rows is None:
        best_rows = greedy_construct(seed=42)

    return best_rows

# Quick self-test
if __name__ == "__main__":
    rows = construct(7, 5)
    diffs = set()
    valid = True
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    max_val = max(max(r) for r in rows)
    print(f"Valid: {valid}, Max: {max_val}")
    print(f"Rows: {rows}")
