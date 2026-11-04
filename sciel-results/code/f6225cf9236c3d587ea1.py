"""Improved program for difference triangle set (7,5) task."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)


def _greedy_row(used, k):
    """Build a single row greedily: start at 0, pick smallest x such that
    all new differences x - a (a in current row) are not in used."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = set()
        conflict = False
        for a in row:
            d = x - a
            if d in used:
                conflict = True
                break
            new_diffs.add(d)
        if not conflict:
            row.append(x)
            used.update(new_diffs)
    return row


def _greedy_full(n_rows, k):
    """Build n_rows rows greedily."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows


def _local_search(rows, k, max_iters=2000, deadline=None):
    """Try to improve scope via local search: pick a row, try to rebuild it
    with a smaller max entry by backtracking, respecting used diffs from other rows."""
    best = rows[:]
    best_scope = _scope(best)
    
    # Precompute diffs for each row
    row_diff_sets = [_row_diffs(r) for r in best]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random row to rebuild
        idx = random.randint(0, len(best) - 1)
        old_row = best[idx]
        old_diffs = row_diff_sets[idx]
        
        # Diffs used by all other rows
        other_diffs = set()
        for j in range(len(best)):
            if j != idx:
                other_diffs |= row_diff_sets[j]
        
        # Try to rebuild row idx with a target max < old_row[-1]
        target_max = old_row[-1] - 1
        if target_max < k:  # need at least k+1 distinct values starting at 0
            continue
        
        # Backtracking search for a new row with max <= target_max
        new_row = _backtrack_row(other_diffs, k, target_max, deadline)
        
        if new_row is not None:
            new_diffs = _row_diffs(new_row)
            # Verify no conflict with other_diffs
            if not (new_diffs & other_diffs):
                best[idx] = new_row
                row_diff_sets[idx] = new_diffs
                new_scope = _scope(best)
                if new_scope < best_scope:
                    best_scope = new_scope
                    # Continue searching for further improvements
                    continue
    
    return best


def _backtrack_row(used_diffs, k, max_val, deadline=None):
    """Backtracking search for a row of k+1 increasing integers starting at 0
    with max <= max_val, such that all differences are not in used_diffs."""
    row = [0]
    
    def _search(current_row, used_local):
        if deadline and time.time() > deadline:
            return None
        if len(current_row) == k + 1:
            return current_row[:]
        
        last = current_row[-1]
        # Try values from last+1 to max_val
        for x in range(last + 1, max_val + 1):
            new_diffs = set()
            conflict = False
            for a in current_row:
                d = x - a
                if d in used_local or d in used_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            if not conflict:
                current_row.append(x)
                used_local.update(new_diffs)
                result = _search(current_row, used_local)
                if result is not None:
                    return result
                current_row.pop()
                used_local.difference_update(new_diffs)
        
        return None
    
    return _search(row, set())


def _simulated_annealing(n_rows, k, deadline):
    """Simulated annealing on row assignments."""
    rows = _greedy_full(n_rows, k)
    best_rows = rows[:]
    best_scope = _scope(best_rows)
    
    row_diff_sets = [_row_diffs(r) for r in best_rows]
    
    start_temp = 50.0
    end_temp = 0.1
    cooling = 0.995
    temp = start_temp
    
    it = 0
    while temp > end_temp and time.time() < deadline:
        it += 1
        # Pick a row to perturb
        idx = random.randint(0, n_rows - 1)
        old_row = best_rows[idx]
        old_diffs = row_diff_sets[idx]
        
        other_diffs = set()
        for j in range(n_rows):
            if j != idx:
                other_diffs |= row_diff_sets[j]
        
        # Perturb: rebuild row with some randomness
        new_row = _random_row(other_diffs, k, old_row[-1] + 5, deadline)
        
        if new_row is None:
            temp *= cooling
            continue
        
        new_diffs = _row_diffs(new_row)
        if new_diffs & other_diffs:
            temp *= cooling
            continue
        
        old_scope = best_scope
        new_scope = max(max(best_rows[j][-1] for j in range(n_rows) if j != idx), new_row[-1])
        
        delta = new_scope - old_scope
        if delta <= 0 or random.random() < __import__('math').exp(-delta / max(temp, 0.01)):
            best_rows[idx] = new_row
            row_diff_sets[idx] = new_diffs
            if new_scope < best_scope:
                best_scope = new_scope
        
        temp *= cooling
    
    return best_rows


def _random_row(used_diffs, k, max_val, deadline):
    """Build a random valid row starting at 0 with max <= max_val."""
    if deadline and time.time() > deadline:
        return None
    row = [0]
    x = 0
    attempts = 0
    while len(row) < k + 1 and attempts < 100:
        if deadline and time.time() > deadline:
            return None
        x += 1
        if x > max_val:
            return None
        new_diffs = set()
        conflict = False
        for a in row:
            d = x - a
            if d in used_diffs:
                conflict = True
                break
            new_diffs.add(d)
        if not conflict:
            row.append(x)
            used_diffs.update(new_diffs)
        else:
            # Sometimes skip ahead
            if random.random() < 0.3:
                x += random.randint(1, 3)
        attempts += 1
    
    if len(row) != k + 1:
        return None
    return row


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all differences distinct across rows."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with greedy solution
    rows = _greedy_full(n_rows, k)
    best_scope = _scope(rows)
    
    # Local search to improve
    if time.time() < deadline - 10.0:
        rows = _local_search(rows, k, max_iters=500, deadline=deadline - 10.0)
        best_scope = _scope(rows)
    
    # Try multiple random restarts with local search
    start_time = time.time()
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Slightly perturb the greedy construction
        used = set()
        rows_r = []
        success = True
        for i in range(n_rows):
            row = _greedy_row(used, k)
            rows_r.append(row)
        
        # Apply local search
        rows_r = _local_search(rows_r, k, max_iters=300, deadline=deadline - 5.0)
        scope_r = _scope(rows_r)
        
        if scope_r < best_scope:
            best_scope = scope_r
            rows = rows_r
    
    # Final local search pass
    if time.time() < deadline - 2.0:
        rows = _local_search(rows, k, max_iters=1000, deadline=deadline - 1.0)
    
    return rows

# EVOLVE-BLOCK-END
