import time
import random
import math
from collections import Counter

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            s.add(row[j] - row[i])
    return s

def _build_diff_counter(rows):
    c = Counter()
    for r in rows:
        for d in _diffs_for_row(r):
            c[d] += 1
    return c

def _collision_count_from_counter(counter):
    return sum(max(0, v - 1) for v in counter.values())

def _fast_greedy(n_rows, k, rng, scope_limit=None):
    """
    Fast greedy construction. For each row, greedily pick the smallest
    value that doesn't create a duplicate difference.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

class SAState:
    def __init__(self, rows):
        self.rows = [r[:] for r in rows]
        self.n_rows = len(rows)
        self.row_len = len(rows[0])
        self.k = self.row_len - 1
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
            for i in range(self.row_len):
                for j in range(i + 1, self.row_len):
                    d = r[j] - r[i]
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

    def _remove_diffs(self, row_idx, pos, val):
        row = self.rows[row_idx]
        for j in range(self.row_len):
            if j != pos:
                d = abs(val - row[j])
                if d in self.diff_count:
                    self.diff_count[d] -= 1
                    if self.diff_count[d] == 1:
                        self.collisions -= 1
                    if self.diff_count[d] == 0:
                        del self.diff_count[d]

    def _add_diffs(self, row_idx, pos, val):
        row = self.rows[row_idx]
        for j in range(self.row_len):
            if j != pos:
                d = abs(val - row[j])
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1

    def _update_max(self, old_val, new_val, row_idx):
        old_max = self.max_val
        if new_val > self.max_val:
            self.max_val = new_val
        elif old_val == old_max and new_val < old_max:
            # Recompute max
            self.max_val = 0
            for r in self.rows:
                if r[-1] > self.max_val:
                    self.max_val = r[-1]

    def try_move(self, r_idx, i_idx, new_val):
        old_val = self.rows[r_idx][i_idx]
        if old_val == new_val:
            return False

        # Check bounds
        if i_idx > 0 and new_val <= self.rows[r_idx][i_idx - 1]:
            return False
        if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx + 1]:
            return False

        # Remove old differences
        self._remove_diffs(r_idx, i_idx, old_val)

        # Apply change
        self.rows[r_idx][i_idx] = new_val
        self._update_max(old_val, new_val, r_idx)

        # Add new differences
        self._add_diffs(r_idx, i_idx, new_val)

        return True

    def undo_move(self, r_idx, i_idx, old_val):
        new_val = self.rows[r_idx][i_idx]
        if new_val == old_val:
            return

        # Remove new differences
        self._remove_diffs(r_idx, i_idx, new_val)

        # Restore old value
        self.rows[r_idx][i_idx] = old_val
        self._update_max(new_val, old_val, r_idx)

        # Add old differences
        self._add_diffs(r_idx, i_idx, old_val)

def _sa_optimize(rows, k, deadline, rng, max_iters=None):
    """
    Simulated Annealing to minimize scope with collision penalty.
    Uses SAState for efficient incremental diff counting.
    """
    if max_iters is None:
        max_iters = 5000000

    state = SAState(rows)
    current_collisions, current_max = state.get_cost()
    best_collisions = current_collisions
    best_max = current_max
    best_rows = [r[:] for r in state.rows]

    collision_weight = 100000

    def cost(coll, max_val):
        return coll * collision_weight + max_val

    T = 50.0
    cooling = 0.9999
    min_T = 0.01

    it = 0
    row_len = k + 1

    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1

        # Choose random row and position
        i = rng.randint(0, state.n_rows - 1)
        j = rng.randint(1, k)

        old_val = state.rows[i][j]
        prev_val = state.rows[i][j - 1]
        next_val = state.rows[i][j + 1] if j < k else old_val + 10000

        # Move type
        move_type = rng.random()

        if move_type < 0.5:
            # Try to reduce (especially if near max)
            upper = old_val - 1 if old_val > prev_val + 1 else prev_val
            if upper <= prev_val:
                continue
            new_val = rng.randint(prev_val + 1, upper)
        elif move_type < 0.8:
            # Small perturbation
            delta = rng.randint(-8, 8)
            new_val = old_val + delta
        else:
            # Larger move
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 30))

        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        if new_val == old_val:
            continue

        # Apply change
        state.rows[i][j] = new_val

        # Check internal validity (all diffs positive and distinct within row)
        row = state.rows[i]
        internal_diffs = set()
        valid_internal = True
        for a_idx in range(row_len):
            for b_idx in range(a_idx + 1, row_len):
                d = row[b_idx] - row[a_idx]
                if d <= 0 or d in internal_diffs:
                    valid_internal = False
                    break
                internal_diffs.add(d)
            if not valid_internal:
                break

        if not valid_internal:
            state.rows[i][j] = old_val
            continue

        # Update diff counter
        for d in _diffs_for_row(row):
            if d in state.diff_count:
                state.diff_count[d] += 1
                if state.diff_count[d] == 2:
                    state.collisions += 1
            else:
                state.diff_count[d] = 1

        # Remove old diffs
        old_row = [old_val if idx == j else row[idx] for idx in range(row_len)]
        for d in _diffs_for_row(old_row):
            if d in state.diff_count:
                state.diff_count[d] -= 1
                if state.diff_count[d] == 1:
                    state.collisions -= 1
                if state.diff_count[d] == 0:
                    del state.diff_count[d]

        new_collisions = state.collisions
        new_scope = state.max_val if new_val > state.max_val else (
            _scope(state.rows) if old_val == state.max_val and new_val < old_val else state.max_val
        )

        old_c = cost(current_collisions, current_max)
        new_c = cost(new_collisions, new_scope)

        if new_c <= old_c:
            accept = True
        else:
            delta_c = new_c - old_c
            prob = math.exp(-delta_c / max(T, min_T))
            accept = rng.random() < prob

        if accept:
            current_collisions = new_collisions
            current_max = new_scope
            if new_scope < best_max or (new_scope == best_max and new_collisions < best_collisions):
                best_max = new_scope
                best_collisions = new_collisions
                best_rows = [r[:] for r in state.rows]
        else:
            # Revert
            state.rows[i][j] = old_val
            for d in _diffs_for_row(state.rows[i]):
                if d in state.diff_count:
                    state.diff_count[d] -= 1
                    if state.diff_count[d] == 1:
                        state.collisions -= 1
                    if state.diff_count[d] == 0:
                        del state.diff_count[d]
            for d in _diffs_for_row(old_row):
                if d in state.diff_count:
                    state.diff_count[d] += 1
                    if state.diff_count[d] == 2:
                        state.collisions += 1
                else:
                    state.diff_count[d] = 1
            current_collisions = state.collisions
            current_max = state.max_val

        T = max(T * cooling, min_T)

    return best_max, best_rows

def _scope_squeeze(rows, k, deadline, rng):
    """
    Try to reduce the scope by 1 iteratively using local search.
    """
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)

    # Precompute other rows' diff sets for faster checking
    other_diff_sets = []
    for i, r in enumerate(rows):
        s = _row_diffs_set(r)
        other_diff_sets.append(s)

    for _ in range(200):
        if time.time() > deadline:
            break

        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break

        i = rng.choice(max_rows)
        row = rows[i]
        old_max = row[-1]
        prev = row[k - 1]

        found = False
        for new_val in range(old_max - 1, prev, -1):
            row[k] = new_val
            new_diffs = _diffs_for_row(row)

            if len(new_diffs) != len(set(new_diffs)):
                continue

            # Check cross-row conflicts
            conflict = False
            new_diffs_set = set(new_diffs)
            for other_idx, other_set in enumerate(other_diff_sets):
                if other_idx == i:
                    continue
                for d in new_diffs_set:
                    if d in other_set:
                        conflict = True
                        break
                if conflict:
                    break

            if not conflict:
                current_scope = new_val
                found = True
                # Update the diff set for this row
                other_diff_sets[i] = new_diffs_set
                break

        if not found:
            row[k] = old_max
            continue

    return rows

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin
    rng = random.Random(12345)

    best_result = None
    best_score = float('inf')

    # Phase 1: Fast greedy baseline
    rows = _fast_greedy(n_rows, k, rng)
    if rows is not None:
        best_result = rows
        best_score = _scope(rows)
    else:
        # Last resort: simple sequential
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_result = rows
        best_score = _scope(rows)

    # Phase 2: Simulated Annealing optimization
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining * 0.75
        max_iters = int(remaining * 50000)
        s, rows = _sa_optimize(best_result, k, sa_deadline, rng, max_iters=max_iters)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows

    # Phase 3: Scope squeeze
    if time.time() < deadline - 3:
        remaining = deadline - time.time()
        squeeze_deadline = time.time() + remaining * 0.8
        squeezed = _scope_squeeze(best_result, k, squeeze_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed

    # Final validation
    if best_result is not None and not _valid(best_result, k):
        # Shouldn't happen, but fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_result = rows

    return best_result
