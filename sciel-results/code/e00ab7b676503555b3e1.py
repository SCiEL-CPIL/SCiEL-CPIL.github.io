"""Difference triangle set (7,5) solver with structured search and aggressive local search."""
import time
import random
import math

_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    """Return the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _build_row_given_used(used, k, rng, max_val=None):
    """Build one row greedily, choosing the smallest available next mark."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            break
        if rng.random() < 0.05:
            x += rng.randint(1, 2)
            if max_val is not None and x > max_val:
                break
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _initial_construction(n_rows, k, rng):
    """Build all rows greedily with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _build_row_given_used(used, k, rng)
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _compute_all_diffs(rows):
    """Compute a dictionary mapping each difference to its count."""
    diff_counts = {}
    for row in rows:
        for d in _row_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
    return diff_counts

def _count_conflicts(diff_counts):
    """Count the number of conflict pairs."""
    c = 0
    for cnt in diff_counts.values():
        if cnt > 1:
            c += cnt - 1
    return c

def _sa_main(rows, k, rng, time_budget=60):
    """Simulated Annealing to minimize the scope (max entry)."""
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    current_rows = [list(r) for r in best_rows]
    
    diff_counts = _compute_all_diffs(current_rows)
    
    def get_scope():
        return _max_entry(current_rows)
    
    current_scope = get_scope()
    current_conflicts = _count_conflicts(diff_counts)
    current_cost = current_scope + 1000 * current_conflicts
    
    best_scope = current_scope
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    start_time = time.time()
    temp = 5.0
    cooling = 0.9995
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
            
        move_type = rng.random()
        
        if move_type < 0.5:
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            
            for d in _row_diffs(old_row):
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            used = set(diff_counts.keys())
            max_limit = current_scope + rng.randint(0, 3)
            new_row = _build_row_given_used(used, k, rng, max_limit)
            
            if len(new_row) < k + 1:
                for d in _row_diffs(old_row):
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                continue
            
            for d in _row_diffs(new_row):
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            current_rows[ri] = new_row
            
            new_scope = get_scope()
            new_conflicts = _count_conflicts(diff_counts)
            new_cost = new_scope + 1000 * new_conflicts
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (temp > 0 and rng.random() < math.exp(-delta_cost / temp)):
                current_scope = new_scope
                current_conflicts = new_conflicts
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = new_scope
                    best_conflicts = new_conflicts
                    best_rows = [list(r) for r in current_rows]
            else:
                for d in _row_diffs(new_row):
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                for d in _row_diffs(old_row):
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    
                current_rows[ri] = old_row

        else:
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            if len(row) < 2:
                continue
            
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            
            if rng.random() < 0.5:
                reduction = rng.randint(1, 2)
                new_val = old_val - reduction
                if new_val <= 0:
                    continue
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
            else:
                delta = rng.randint(-2, 2)
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
            
            if new_val == old_val:
                continue
                
            old_diffs = _row_diffs(row)
            
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            row[idx] = new_val
            new_row = row
            new_diffs = _row_diffs(new_row)
            
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_scope = get_scope()
            new_conflicts = _count_conflicts(diff_counts)
            new_cost = new_scope + 1000 * new_conflicts
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (temp > 0 and rng.random() < math.exp(-delta_cost / temp)):
                current_scope = new_scope
                current_conflicts = new_conflicts
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = new_scope
                    best_conflicts = new_conflicts
                    best_rows = [list(r) for r in current_rows]
            else:
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    
                row[idx] = old_val
                current_rows[ri][idx] = old_val

        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    if _is_valid(best_rows, k):
        return best_rows
    else:
        if _is_valid(current_rows, k):
            return current_rows
        return best_rows

def _aggressive_polish(rows, k, rng, time_budget=5):
    """Final high-precision polish: try single-entry shifts to reduce scope."""
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    best_cost = best_scope
    
    start_time = time.time()
    
    all_diffs = set()
    for row in best_rows:
        for d in _row_diffs(row):
            all_diffs.add(d)
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 3:
            break
        
        improved = False
        for ri in range(n_rows):
            row = best_rows[ri]
            for idx in range(1, len(row)):
                old_val = row[idx]
                new_val = old_val - 1
                if new_val <= 0:
                    continue
                if idx > 1 and new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
                
                old_diffs = _row_diffs(row)
                row[idx] = new_val
                new_diffs = _row_diffs(row)
                
                temp_diffs = set(all_diffs)
                for d in old_diffs:
                    temp_diffs.discard(d)
                conflict = False
                for d in new_diffs:
                    if d in temp_diffs:
                        conflict = True
                        break
                    temp_diffs.add(d)
                
                if not conflict:
                    all_diffs = temp_diffs
                    new_scope = _max_entry(best_rows)
                    new_cost = new_scope
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_scope = new_scope
                        improved = True
                else:
                    row[idx] = old_val
        
        if not improved:
            break
    
    if _is_valid(best_rows, k):
        return best_rows
    return [list(r) for r in rows]

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    best_rows = None
    best_score = float('inf')
    
    n_restarts = 150
    for i in range(n_restarts):
        if _elapsed() > _DEADLINE - 20:
            break
        rng2 = random.Random(12345 + i * 7919)
        rows = _initial_construction(n_rows, k, rng2)
        if rows is None:
            continue
        if not _is_valid(rows, k):
            continue
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_score = _max_entry(best_rows)
    
    if _elapsed() < _DEADLINE - 15:
        time_left = _DEADLINE - 15 - _elapsed()
        sa_time = min(80, time_left)
        improved_rows = _sa_main(best_rows, k, rng, time_budget=sa_time)
        
        if improved_rows and _is_valid(improved_rows, k):
            new_score = _max_entry(improved_rows)
            if new_score < best_score:
                best_rows = improved_rows
                best_score = new_score
    
    if _elapsed() < _DEADLINE - 5:
        polish_time = min(5, _DEADLINE - 5 - _elapsed())
        if polish_time > 0:
            polished_rows = _aggressive_polish(best_rows, k, rng, time_budget=polish_time)
            if polished_rows and _is_valid(polished_rows, k):
                new_score = _max_entry(polished_rows)
                if new_score < best_score:
                    best_rows = polished_rows
                    best_score = new_score
    
    if not _is_valid(best_rows, k):
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    return best_rows
