"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Quick greedy to get a valid baseline
    baseline = _greedy(n_rows, k)
    best = baseline
    best_max = max(max(r) for r in best)
    
    # Try to improve with local search
    if time.time() < deadline - 10:
        improved = _local_search(n_rows, k, best, best_max, deadline)
        if improved is not None:
            best = improved
            best_max = max(max(r) for r in best)
    
    # Try targeted backtracking for smaller bounds
    if time.time() < deadline - 5:
        for target in range(best_max - 1, best_max - 8, -1):
            if time.time() > deadline - 5:
                break
            sol = _backtrack_solve(n_rows, k, target, deadline)
            if sol is not None:
                best = sol
                best_max = target
                break
    
    return best


def _greedy(n_rows, k):
    """Greedy construction: each row takes smallest next mark with unused differences."""
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


def _get_all_diffs(rows):
    """Get all positive differences from all rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows, n_rows, k):
    """Check if rows form a valid difference triangle set."""
    if len(rows) != n_rows:
        return False
    diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k + 1):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True


def _local_search(n_rows, k, initial_rows, initial_max, deadline):
    """Simulated annealing / local search to minimize max entry."""
    rows = [list(r) for r in initial_rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    current_max = max(max(r) for r in rows)
    T = 50.0
    cooling = 0.995
    min_T = 0.5
    
    while time.time() < deadline - 10 and T > min_T:
        # Pick a random row and position to perturb
        r_idx = random.randint(0, n_rows - 1)
        # Perturb one element (not the first)
        pos = random.randint(1, k)
        old_val = rows[r_idx][pos]
        
        # Try changing to a nearby value
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        if new_val <= 0:
            continue
        
        # Check if row remains strictly increasing
        if pos > 0 and new_val <= rows[r_idx][pos - 1]:
            continue
        if pos < k and new_val >= rows[r_idx][pos + 1]:
            continue
        
        rows[r_idx][pos] = new_val
        
        # Check validity
        if _is_valid(rows, n_rows, k):
            new_max = max(max(r) for r in rows)
            if new_max < current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]
            else:
                # Accept with some probability
                if random.random() < math.exp(-(new_max - current_max) / T):
                    current_max = new_max
                else:
                    rows[r_idx][pos] = old_val
        else:
            rows[r_idx][pos] = old_val
        
        T *= cooling
    
    if best_max < initial_max:
        return best_rows
    return None


def _backtrack_solve(n_rows, k, max_val, deadline):
    """Try to find a valid solution with all entries <= max_val using backtracking."""
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, row):
        if time.time() > deadline:
            return False
        if row_idx == n_rows:
            return True
        
        # Try to build this row
        if not _build_row(row_idx, row, 0, used_diffs, max_val, deadline):
            return False
        
        rows.append(row)
        if backtrack(row_idx + 1, [0]):
            return True
        rows.pop()
        return False
    
    def _build_row(row_idx, row, pos, used_diffs, max_val, deadline):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        
        if pos == 0:
            row.append(0)
            return _build_row(row_idx, row, 1, used_diffs, max_val, deadline)
        
        start = row[-1] + 1
        for val in range(start, max_val + 1):
            new_diffs = {val - a for a in row}
            if not new_diffs & used_diffs:
                row.append(val)
                used_diffs_copy = used_diffs | new_diffs
                if _build_row(row_idx, row, pos + 1, used_diffs_copy, max_val, deadline):
                    return True
                row.pop()
        
        return False
    
    if backtrack(0, [0]):
        return rows
    return None
# EVOLVE-BLOCK-END
