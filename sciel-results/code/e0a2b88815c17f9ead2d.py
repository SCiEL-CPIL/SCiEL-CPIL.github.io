import time
import random
import math
import sys
from collections import defaultdict


def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    # Hard upper bound for any value we'll consider
    HARD_BOUND = 120

    def greedy_construct():
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

    # Get baseline
    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(max(r) for r in best_rows)

    def compute_cost(rows):
        used = {}
        max_val = 0
        for r in rows:
            if r[-1] > max_val:
                max_val = r[-1]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        used[d] += 1
                    else:
                        used[d] = 1
        collisions = 0
        for cnt in used.values():
            if cnt > 1:
                collisions += cnt - 1
        return collisions, max_val

    # State class for fast incremental updates
    class SAState:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self._rebuild()

        def _rebuild(self):
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            for r in self.rows:
                if r[-1] > self.max_val:
                    self.max_val = r[-1]
                for i in range(len(r)):
                    ri = r[i]
                    for j in range(i + 1, len(r)):
                        d = r[j] - ri
                        if d in self.diff_count:
                            self.diff_count[d] += 1
                            if self.diff_count[d] == 2:
                                self.collisions += 1
                        else:
                            self.diff_count[d] = 1

        def get_cost(self):
            return self.collisions, self.max_val

        def is_valid(self):
            return self.collisions == 0

        def try_move(self, r_idx, i_idx, new_val):
            """Change rows[r_idx][i_idx] to new_val. Return True if applied."""
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return False
            row = self.rows[r_idx]
            if i_idx > 0 and new_val <= row[i_idx - 1]:
                return False
            if i_idx < self.row_len - 1 and new_val >= row[i_idx + 1]:
                return False
            if new_val < 0 or new_val > HARD_BOUND:
                return False

            # Remove old differences
            dc = self.diff_count
            for j in range(self.row_len):
                if j == i_idx:
                    continue
                d = abs(old_val - row[j])
                c = dc[d] - 1
                if c == 1:
                    self.collisions -= 1
                if c == 0:
                    del dc[d]
                else:
                    dc[d] = c

            old_max = self.max_val
            row[i_idx] = new_val

            # Add new differences
            for j in range(self.row_len):
                if j == i_idx:
                    continue
                d = abs(new_val - row[j])
                if d in dc:
                    c = dc[d] + 1
                    dc[d] = c
                    if c == 2:
                        self.collisions += 1
                else:
                    dc[d] = 1

            if new_val > self.max_val:
                self.max_val = new_val
            elif old_val == old_max and new_val < old_max:
                self.max_val = max(max(r) for r in self.rows)
            return True

        def undo_move(self, r_idx, i_idx, old_val):
            """Restore rows[r_idx][i_idx] to old_val."""
            new_val = self.rows[r_idx][i_idx]
            if new_val == old_val:
                return
            row = self.rows[r_idx]
            dc = self.diff_count
            for j in range(self.row_len):
                if j == i_idx:
                    continue
                d = abs(new_val - row[j])
                c = dc[d] - 1
                if c == 1:
                    self.collisions -= 1
                if c == 0:
                    del dc[d]
                else:
                    dc[d] = c

            row[i_idx] = old_val

            for j in range(self.row_len):
                if j == i_idx:
                    continue
                d = abs(old_val - row[j])
                if d in dc:
                    c = dc[d] + 1
                    dc[d] = c
                    if c == 2:
                        self.collisions += 1
                else:
                    dc[d] = 1

            if old_val > self.max_val:
                self.max_val = old_val
            elif new_val == self.max_val and old_val < new_val:
                self.max_val = max(max(r) for r in self.rows)

    def soft_row_weights(state, beta=0.05):
        """Weights over rows proportional to exp(beta * row_max)."""
        vals = [r[-1] for r in state.rows]
        m = max(vals)
        ws = [math.exp(beta * (v - m)) for v in vals]
        tot = sum(ws)
        return [w / tot for w in ws]

    def pick_row(state, beta=0.05):
        ws = soft_row_weights(state, beta)
        return random.choices(range(state.n_rows), weights=ws, k=1)[0]

    def local_polish(rows, time_limit):
        """High-precision polish: keep validity, compress the scope."""
        state = SAState(rows)
        current_collisions, current_max = state.get_cost()
        best_collisions = current_collisions
        best_max = current_max
        best_rows = [r[:] for r in state.rows]
        if best_collisions > 0:
            return best_rows, best_max, best_collisions

        start = time.time()
        beta = 0.06
        beta_floor = 0.008
        beta_cool = 0.9995

        while time.time() - start < time_limit:
            r_idx = pick_row(state, beta)
            row = state.rows[r_idx]

            # Bias position choice toward the tail
            if random.random() < 0.65:
                i_idx = random.randint(max(1, row_len - 3), row_len - 1)
            else:
                i_idx = random.randint(1, row_len - 1)

            lower = row[i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = row[i_idx + 1] - 1
            else:
                # For the last element, try to reduce it
                upper = row[i_idx] - 1

            if lower > upper:
                continue

            old_val = row[i_idx]
            # Candidate set: tightest feasible plus random values
            cands = set()
            cands.add(lower)
            span = upper - lower + 1
            for _ in range(5):
                cands.add(random.randint(lower, upper))
            if old_val in cands:
                cands.discard(old_val)

            for new_val in cands:
                if new_val == old_val:
                    continue
                if not state.try_move(r_idx, i_idx, new_val):
                    continue
                # Only accept collision-free moves that do not worsen scope
                if state.collisions == 0 and state.max_val <= best_max:
                    if state.max_val < best_max:
                        best_max = state.max_val
                        best_rows = [r[:] for r in state.rows]
                else:
                    state.undo_move(r_idx, i_idx, old_val)

            beta *= beta_cool
            if beta < beta_floor:
                beta = beta_floor

        return best_rows, best_max, best_collisions

    def simulate_annealing(initial_rows, time_limit):
        state = SAState(initial_rows)
        current_collisions, current_max = state.get_cost()
        best_collisions = current_collisions
        best_max = current_max
        best_rows = [r[:] for r in state.rows]

        T = 50.0
        cooling = 0.9998
        start = time.time()

        while time.time() - start < time_limit:
            if state.collisions == 0 and random.random() < 0.5:
                r_idx = pick_row(state, 0.04)
            else:
                r_idx = random.randint(0, n_rows - 1)
            if random.random() < 0.65:
                i_idx = random.randint(max(1, row_len - 3), row_len - 1)
            else:
                i_idx = random.randint(1, row_len - 1)

            lower = state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = state.rows[r_idx][i_idx + 1] - 1
            else:
                upper = min(HARD_BOUND, best_max + 15)

            if lower > upper:
                continue

            for _ in range(3):
                new_val = random.randint(lower, upper)
                old_val = state.rows[r_idx][i_idx]
                if new_val == old_val:
                    continue

                if not state.try_move(r_idx, i_idx, new_val):
                    continue

                new_collisions, new_max = state.get_cost()

                if new_collisions < current_collisions:
                    accept = True
                elif new_collisions == current_collisions:
                    if new_max < current_max:
                        accept = True
                    else:
                        delta = new_max - current_max
                        if delta < 0:
                            accept = True
                        else:
                            p = math.exp(-delta / max(T, 0.1))
                            accept = random.random() < p
                else:
                    delta = (new_collisions - current_collisions) * 100 + (new_max - current_max)
                    if delta < 0:
                        accept = True
                    else:
                        p = math.exp(-delta / max(T, 0.1))
                        accept = random.random() < p

                if accept:
                    current_collisions = new_collisions
                    current_max = new_max
                    if current_collisions < best_collisions or (
                        current_collisions == best_collisions and current_max < best_max
                    ):
                        best_collisions = current_collisions
                        best_max = current_max
                        best_rows = [r[:] for r in state.rows]
                else:
                    state.undo_move(r_idx, i_idx, old_val)

            T *= cooling
            if T < 0.5:
                T = 0.5

        return best_rows, best_max, best_collisions

    # Main loop with explicit budget split
    sa_end = start_time + 120.0
    while time.time() < sa_end:
        if random.random() < 0.3:
            init_rows = [r[:] for r in best_rows]
        else:
            bound = min(HARD_BOUND, best_max + 8)
            init_rows = []
            for _ in range(n_rows):
                if bound < k:
                    pts = list(range(1, k + 1))
                else:
                    pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)

        sa_time = min(8.0, sa_end - time.time())
        if sa_time < 1.0:
            break

        local_rows, local_max, local_collisions = simulate_annealing(init_rows, sa_time)

        best_coll, best_mx = compute_cost(best_rows)
        if local_collisions < best_coll or (local_collisions == 0 and local_max < best_max):
            best_rows = local_rows
            best_max = local_max

        # If valid, try to polish (interleaved)
        if compute_cost(best_rows)[0] == 0 and time.time() < sa_end - 5.0:
            polish_time = min(3.0, sa_end - time.time())
            polished_rows, polished_max, _ = local_polish(best_rows, polish_time)
            if polished_max < best_max:
                best_rows = polished_rows
                best_max = polished_max

    # Final high-precision polish pass
    if compute_cost(best_rows)[0] == 0 and time.time() < deadline - 3.0:
        polish_time = min(22.0, deadline - time.time() - 3.0)
        if polish_time > 0.5:
            polished_rows, polished_max, _ = local_polish(best_rows, polish_time)
            if polished_max < best_max:
                best_rows = polished_rows
                best_max = polished_max

    # One more polish iteration if time allows
    if compute_cost(best_rows)[0] == 0 and time.time() < deadline - 1.0:
        polish_time = min(10.0, deadline - time.time() - 1.0)
        if polish_time > 0.5:
            polished_rows, polished_max, _ = local_polish(best_rows, polish_time)
            if polished_max < best_max:
                best_rows = polished_rows
                best_max = polished_max

    return best_rows
