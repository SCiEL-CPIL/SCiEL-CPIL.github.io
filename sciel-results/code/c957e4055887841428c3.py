"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs(row):
        """Get set of all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ai)
        return diffs

    def all_diffs(rows):
        """Get the union of all differences across all rows."""
        used = set()
        for row in rows:
            used |= row_diffs(row)
        return used

    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        used = set()
        for row in rows:
            for d in row_diffs(row):
                if d in used:
                    return False
                used.add(d)
        return True

    def scope(rows):
        """Return the maximum entry across all rows."""
        return max(max(row) for row in rows)

    def greedy_row(used, k, max_scope=None, rng=None, skip_prob=0.0):
        """Build one row greedily, starting at 0, with k+1 entries.
        Returns (row, success)."""
        row = [0]
        current_diffs = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            if max_scope is not None and x > max_scope:
                return None, False
            new = {x - a for a in row}
            if new & used or new & current_diffs:
                continue
            if rng is not None and skip_prob > 0 and rng.random() < skip_prob:
                continue
            row.append(x)
            current_diffs |= new
            used |= new
        return row, True

    def greedy_all(n_rows, k, rng=None, skip_prob=0.0, max_scope=None):
        """Build all rows greedily."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row, ok = greedy_row(used, k, max_scope, rng, skip_prob)
            if not ok:
                return None
            rows.append(row)
        return rows

    def backtracking(n_rows, k, used, rows, row_idx, max_scope, rng, time_limit):
        """Backtracking search to build remaining rows."""
        if time.time() - start_time > time_limit:
            return False
        if row_idx == n_rows:
            return True
        # Try to build row row_idx with backtracking
        row = [0]
        current_diffs = set()
        
        def bt(pos, x):
            if time.time() - start_time > time_limit:
                return False
            if pos == k + 1:
                rows.append(row[:])
                return backtracking(n_rows, k, used, rows, row_idx + 1, max_scope, rng, time_limit)
            # Try candidates from x+1 upward
            for trial in range(x + 1, max_scope + 1):
                new = {trial - a for a in row}
                if new & used or new & current_diffs:
                    continue
                # Check if this leads to a dead end quickly (heuristic)
                row.append(trial)
                current_diffs |= new
                used |= new
                if bt(pos + 1, trial):
                    return True
                row.pop()
                current_diffs -= new
                used -= new
            return False
        
        if bt(1, 0):
            return True
        return False

    # Phase 1: Fast deterministic greedy baseline
    rows = greedy_all(n_rows, k)
    best_rows = rows if rows else None
    best_scope = scope(best_rows) if best_rows else float('inf')

    if time_left() < 5:
        return best_rows

    # Phase 2: Randomized greedy with restarts
    rng = random.Random(42)
    for restart in range(500):
        if time_left() < 10:
            break
        # Vary skip probability
        skip_prob = rng.random() * 0.3
        # Occasionally try a tighter max_scope
        max_scope = best_scope - 1 if best_scope < float('inf') else None
        rows = greedy_all(n_rows, k, rng, skip_prob, max_scope)
        if rows and is_valid(rows):
            s = scope(rows)
            if s < best_scope:
                best_rows = rows
                best_scope = s

    if time_left() < 10:
        return best_rows

    # Phase 3: Local search - rebuild each row with lower max
    improved = True
    while improved and time_left() > 15:
        improved = False
        for i in range(n_rows):
            if time_left() < 15:
                break
            # Compute differences from other rows
            used_other = set()
            for j in range(n_rows):
                if j != i:
                    used_other |= row_diffs(best_rows[j])
            
            current_max = max(best_rows[i])
            # Try to find a row with max < current_max
            for target_max in range(current_max - 1, 0, -1):
                if time_left() < 15:
                    break
                new_row, ok = greedy_row(set(used_other), k, target_max, rng, 0.0)
                if ok:
                    test_rows = [list(r) for r in best_rows]
                    test_rows[i] = new_row
                    if is_valid(test_rows):
                        best_rows[i] = new_row
                        best_scope = scope(best_rows)
                        improved = True
                        break

    if time_left() < 10:
        return best_rows

    # Phase 4: Backtracking for small improvements
    for attempt in range(20):
        if time_left() < 20:
            break
        target_scope = best_scope - 1
        if target_scope < k + 1:
            break
        used = set()
        rows = []
        time_limit = time_left() - 20
        if backtracking(n_rows, k, used, rows, 0, target_scope, rng, time_limit):
            if is_valid(rows):
                s = scope(rows)
                if s < best_scope:
                    best_rows = rows
                    best_scope = s

    return best_rows
# EVOLVE-BLOCK-END
