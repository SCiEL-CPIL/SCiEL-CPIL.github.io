"""Rewritten program: structural shift from greedy construction to constraint-satisfaction search.
Uses randomized greedy with compaction + local search (simulated annealing) over row permutations
and targeted re-insertion to find tighter difference-triangle sets."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_scope = float('inf')
    best_rows = None

    # Precompute total number of differences needed
    # Each row of k+1 entries has C(k+1, 2) positive differences
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    # Helper: compute all positive differences in a row
    def row_diffs(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]

    # Helper: check if a set of differences is all distinct (no duplicates)
    def is_valid_diffs(diffs):
        return len(diffs) == len(set(diffs))

    # Greedy construction with randomization
    def greedy_random(seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a window of candidates, pick randomly among valid ones for diversity
                found = False
                # Limit search window to keep it fast
                max_x = x + 200
                candidates = []
                while x < max_x:
                    x += 1
                    new = [x - a for a in row]
                    if not any(d in used for d in new):
                        candidates.append(x)
                        if len(candidates) >= 10:
                            break
                if not candidates:
                    # If no candidate in window, extend
                    while True:
                        x += 1
                        new = [x - a for a in row]
                        if not any(d in used for d in new):
                            candidates.append(x)
                            break
                # Pick randomly from candidates for diversity
                x = random.choice(candidates)
                new = [x - a for a in row]
                row.append(x)
                used.update(new)
            rows.append(row)
        return rows

    # Compaction: try to reduce the scope by shifting entries left
    def compact(rows):
        """Try to compact each row by removing gaps where possible."""
        changed = True
        while changed:
            changed = False
            for r in range(len(rows)):
                row = rows[r]
                # Try to shift each entry (except 0) to the left
                for i in range(1, len(row)):
                    original = row[i]
                    # Try values from original-1 down to row[i-1]+1
                    for val in range(original - 1, row[i-1], -1):
                        # Check if this value creates duplicate differences
                        new_diffs = set()
                        for j in range(len(row)):
                            if j != i:
                                new_diffs.add(abs(row[j] - val))
                        # Also need to check against all other rows
                        valid = True
                        for other_row in rows:
                            if other_row is row:
                                continue
                            for d in new_diffs:
                                # Check if d already exists in other_row
                                for a in range(len(other_row)):
                                    for b in range(a + 1, len(other_row)):
                                        if other_row[b] - other_row[a] == d:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                            if not valid:
                                break
                        if valid:
                            # Also check within same row (excluding i)
                            for a in range(len(row)):
                                if a == i:
                                    continue
                                for b in range(a + 1, len(row)):
                                    if a == i or b == i:
                                        continue
                                    if row[b] - row[a] == d:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                row[i] = val
                                changed = True
                                break
        return rows

    # Faster validity check using a set
    def get_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        diffs = get_all_diffs(rows)
        return len(diffs) == n_rows * (k + 1) * k // 2

    def scope(rows):
        return max(max(row) for row in rows)

    # Local search: swap entries between rows or shift within a row
    def local_search(rows, time_limit=10.0):
        """Simulated annealing style local search."""
        start = time.time()
        current = [row[:] for row in rows]
        current_scope = scope(current)
        best = [row[:] for row in current]
        best_scope = current_scope
        temperature = max(current_scope * 0.1, 5.0)
        
        while time.time() - start < time_limit:
            # Pick a random move
            move_type = random.random()
            r = random.randint(0, len(current) - 1)
            i = random.randint(1, len(current[r]) - 1)
            
            if move_type < 0.5:
                # Shift an entry
                delta = random.choice([-1, 1, -2, 2])
                new_val = current[r][i] + delta
                if new_val <= current[r][i-1] or (i < len(current[r])-1 and new_val >= current[r][i+1]):
                    continue
                old_val = current[r][i]
                current[r][i] = new_val
                if is_valid(current):
                    new_scope = scope(current)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if new_scope < best_scope:
                            best = [row[:] for row in current]
                            best_scope = new_scope
                    else:
                        current[r][i] = old_val
                else:
                    current[r][i] = old_val
            else:
                # Try to insert a different value
                new_val = random.randint(current[r][i-1] + 1, current[r][i] + 5)
                if i < len(current[r]) - 1 and new_val >= current[r][i+1]:
                    continue
                if new_val == current[r][i]:
                    continue
                old_val = current[r][i]
                current[r][i] = new_val
                if is_valid(current):
                    new_scope = scope(current)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        if new_scope < best_scope:
                            best = [row[:] for row in current]
                            best_scope = new_scope
                    else:
                        current[r][i] = old_val
                else:
                    current[r][i] = old_val
            
            temperature *= 0.999
        
        return best, best_scope

    # Multi-start greedy + local search
    attempts = 0
    while time.time() < deadline - 5.0:
        rows = greedy_random(seed=None)
        rows = compact(rows)
        if is_valid(rows):
            s = scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [row[:] for row in rows]
        
        # Do local search on the best found so far
        if best_rows is not None and time.time() < deadline - 20.0:
            improved, imp_scope = local_search(best_rows, time_limit=min(10.0, max(1.0, deadline - 25.0 - time.time())))
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
        
        attempts += 1
        # Limit total attempts to avoid infinite loop
        if attempts > 50:
            break

    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = [x - a for a in row]
                if not any(d in used for d in new):
                    row.append(x)
                    used.update(new)
            rows.append(row)
        return rows

    return best_rows
# EVOLVE-BLOCK-END
