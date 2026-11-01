"""Rewritten program for difference triangle set (4,7)."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_set(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            row_diffs = set()
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Backtracking with constraint propagation ----------
    def backtrack_row(used_other, max_span, num_restarts=3, time_limit=None):
        """Build a single row using backtracking with forward checking."""
        if time_limit is None:
            time_limit = 5.0
        
        best_row = None
        best_max = float('inf')
        
        for restart in range(num_restarts):
            if time_left() < 2:
                break
            
            row = [0]
            current_diffs = set()
            start_x = random.randint(1, max(1, max_span // (k + 1)))
            
            # Use a stack for backtracking
            stack = [(start_x,)]
            visited_states = set()
            
            def dfs(row, current_diffs, depth, max_depth=200):
                nonlocal best_row, best_max
                if time_left() < 1:
                    return False
                if len(row) > max_depth:
                    return False
                
                if len(row) == k + 1:
                    if max(row) < best_max:
                        best_max = max(row)
                        best_row = list(row)
                    return True
                
                # Forward checking: try next values
                last = row[-1]
                for trial_x in range(last + 1, max_span + 1):
                    new = {trial_x - a for a in row}
                    if new & used_other or new & current_diffs:
                        continue
                    
                    # Check if remaining entries can still fit
                    remaining = (k + 1) - len(row)
                    if remaining > 0 and trial_x + remaining > max_span:
                        break
                    
                    new_row = row + [trial_x]
                    new_diffs = current_diffs | new
                    
                    if dfs(new_row, new_diffs, depth + 1, max_depth):
                        return True
                    
                    if time_left() < 2:
                        return False
                    if depth > 150:
                        return False
                
                return False
            
            # Try multiple starting points
            for start in [start_x, 1, max_span // (k + 1), max_span // (k + 1) + random.randint(-2, 2)]:
                if start < 1 or start > max_span:
                    continue
                row = [0, start]
                current_diffs = {start}
                if current_diffs & used_other:
                    continue
                if dfs(row, current_diffs, 0):
                    break
                if time_left() < 2:
                    break
        
        return best_row

    def try_reduce_row(rows, row_idx, used_other, target_max, num_restarts=5):
        """Try to find a row at row_idx with max entry <= target_max."""
        for restart in range(num_restarts):
            if time_left() < 3:
                break
            row = backtrack_row(used_other, target_max, num_restarts=1, time_limit=3.0)
            if row is not None and max(row) <= target_max:
                return row
        return None

    def local_polish(rows, max_passes=10, time_threshold=10):
        """Reduce scope by locally re-optimizing each row."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time_left() > time_threshold and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Process rows in order of largest scope first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < time_threshold:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                
                current_max = max(rows[i])
                # Try to reduce by 1, 2, 3...
                for delta in range(1, 10):
                    if time_left() < time_threshold:
                        break
                    target = current_max - delta
                    if target < k:
                        break
                    new_row = try_reduce_row(rows, i, used_other, target, num_restarts=3)
                    if new_row is not None:
                        rows[i] = new_row
                        current_max = max(new_row)
                        improved = True
                        break
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)

        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])

            # Try to rebuild with a slightly larger span to allow exploration
            success = False
            for attempt in range(10):
                max_span = max(rows[idx]) + 20 + attempt * 5
                row = backtrack_row(used_other, max_span, num_restarts=2, time_limit=2.0)
                if row is not None:
                    rows[idx] = row
                    success = True
                    break

            if not success:
                # Fallback to greedy
                greedy_row = [0]
                row_diffs = set()
                x = 0
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # Initial polish
    if time_left() > 20:
        polished = local_polish(best_rows, max_passes=8, time_threshold=15)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    while time_left() > 30:
        # 1. Local Search
        polished = local_polish(best_rows, max_passes=6, time_threshold=15)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        # 2. Kick
        if time_left() < 25:
            break
        kicked_rows = kick(best_rows)

        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows, max_passes=6, time_threshold=15)

        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final high-precision polish
    if time_left() > 10:
        polished = local_polish(best_rows, max_passes=15, time_threshold=5)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
