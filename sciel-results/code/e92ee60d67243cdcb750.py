import time
import sys
import random

def _row_diffs(row):
    """Return the set of all positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _scope_of(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows) if rows else 0

def _validate(rows, n_rows, k):
    """Check if the solution is valid."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        row_d = _row_diffs(row)
        if row_d & used:
            return False
        used |= row_d
    return True

def _pure_greedy(n_rows, k, seed=0):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _randomized_greedy(n_rows, k, seed=42, bias=0.3, max_scope=200):
    """Randomized greedy with occasional larger jumps."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            best_x = None
            candidates = []
            for try_x in range(x + 1, min(x + 50, max_scope + 1)):
                new = {try_x - a for a in row}
                if not (new & used):
                    candidates.append(try_x)
            if not candidates:
                try_x = x + 1
                while True:
                    new = {try_x - a for a in row}
                    if not (new & used):
                        candidates.append(try_x)
                        break
                    try_x += 1
            
            if candidates:
                if rng.random() < bias and len(candidates) > 1:
                    best_x = rng.choice(candidates)
                else:
                    best_x = candidates[0]
            
            if best_x is not None:
                x = best_x
                row.append(x)
                used |= {x - a for a in row[:-1]}
            else:
                break
        rows.append(row)
    return rows

def _lookahead_greedy(n_rows, k, rng):
    """
    Greedy with lookahead: when choosing a value, consider how many
    future valid options it leaves for subsequent positions.
    """
    used = set()
    rows = []
    for row_idx in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            valid_candidates = []
            search_limit = x + 60
            for candidate in range(x + 1, search_limit + 1):
                new_diffs = {candidate - a for a in row}
                if not new_diffs & used:
                    valid_candidates.append(candidate)
                    if len(valid_candidates) >= 10:
                        break
            if not valid_candidates:
                candidate = x + 1
                while True:
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append(candidate)
                        break
                    candidate += 1

            # Score each candidate by how many options it leaves for next step
            best_score = -1
            best_cand = valid_candidates[0]
            for cand in valid_candidates[:8]:
                new_diffs = {cand - a for a in row}
                next_count = 0
                for next_cand in range(cand + 1, cand + 30):
                    next_diffs = {next_cand - a for a in row}
                    next_diffs.add(next_cand - cand)
                    if not (next_diffs & (used | new_diffs)):
                        next_count += 1
                    if next_count >= 5:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand

            x = best_cand
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows

def _hill_climbing(n_rows, k, seed=0, time_budget=30.0):
    """Local search: start from greedy, then try to reduce scope via swaps."""
    start_time = time.time()
    rng = random.Random(seed)
    
    # Start with greedy
    best_rows = _pure_greedy(n_rows, k)
    best_scope = _scope_of(best_rows)
    
    def get_used(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used
    
    def try_improve(rows, scope_limit, rng, max_iters=1000):
        """Try to find a valid solution with max scope <= scope_limit via local search."""
        used = get_used(rows)
        
        for _ in range(max_iters):
            if time.time() - start_time > time_budget * 0.8:
                return None
            
            row_idx = rng.randrange(n_rows)
            row = rows[row_idx]
            
            if len(row) < 2:
                continue
                
            last_idx = len(row) - 1
            old_val = row[last_idx]
            
            if old_val <= scope_limit:
                continue
            
            row_d = _row_diffs(row)
            
            found = False
            for new_val in range(scope_limit, row[last_idx - 1], -1):
                if new_val == old_val:
                    continue
                if new_val <= row[last_idx - 1]:
                    continue
                
                new_row = row[:-1] + [new_val]
                new_row_d = _row_diffs(new_row)
                
                removed = row_d - new_row_d
                added = new_row_d - row_d
                
                other_used = (used - row_d) | (row_d & new_row_d)
                if added & other_used:
                    continue
                
                rows[row_idx] = new_row
                used = get_used(rows)
                found = True
                break
            
            if not found:
                pos = rng.randrange(1, len(row) - 1)
                old_val = row[pos]
                
                if old_val <= scope_limit:
                    continue
                
                row_d = _row_diffs(row)
                found = False
                for new_val in range(scope_limit, row[pos - 1], -1):
                    if new_val == old_val:
                        continue
                    if new_val <= row[pos - 1] or new_val >= row[pos + 1]:
                        continue
                    
                    new_row = row[:pos] + [new_val] + row[pos+1:]
                    new_row_d = _row_diffs(new_row)
                    
                    removed = row_d - new_row_d
                    added = new_row_d - row_d
                    
                    other_used = (used - row_d) | (row_d & new_row_d)
                    if added & other_used:
                        continue
                    
                    rows[row_idx] = new_row
                    used = get_used(rows)
                    found = True
                    break
        
        return rows
    
    current_scope = best_scope
    for target_scope in range(current_scope - 1, max(50, current_scope - 20), -1):
        candidate_rows = [list(r) for r in best_rows]
        result = try_improve(candidate_rows, target_scope, rng)
        if result is not None and _validate(result, n_rows, k):
            new_scope = _scope_of(result)
            if new_scope < best_scope:
                best_rows = result
                best_scope = new_scope
                current_scope = new_scope
    
    return best_rows

def construct(n_rows, k):
    """Construct difference triangle set with minimum scope."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # First, get a baseline with pure greedy
    baseline = _pure_greedy(n_rows, k)
    best_rows = baseline
    best_scope = _scope_of(baseline)
    
    if time.time() > deadline - 10:
        return best_rows
    
    # Try randomized greedy with different seeds
    for seed in range(15):
        if time.time() > deadline - 35:
            break
        try:
            candidate = _randomized_greedy(n_rows, k, seed=seed, bias=0.2 + seed * 0.04, max_scope=best_scope + 30)
            if _validate(candidate, n_rows, k):
                scope = _scope_of(candidate)
                if scope < best_scope:
                    best_rows = candidate
                    best_scope = scope
        except Exception:
            continue
    
    # Try lookahead greedy with a few seeds
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        try:
            rng = random.Random(seed * 1000 + 7)
            candidate = _lookahead_greedy(n_rows, k, rng)
            if _validate(candidate, n_rows, k):
                scope = _scope_of(candidate)
                if scope < best_scope:
                    best_rows = candidate
                    best_scope = scope
        except Exception:
            continue
    
    if time.time() > deadline - 5:
        return best_rows
    
    # Try hill climbing to further reduce
    for seed in range(5):
        if time.time() > deadline - 5:
            break
        try:
            time_budget = min(20.0, max(5.0, deadline - time.time() - 5))
            candidate = _hill_climbing(n_rows, k, seed=seed, time_budget=time_budget)
            if _validate(candidate, n_rows, k):
                scope = _scope_of(candidate)
                if scope < best_scope:
                    best_rows = candidate
                    best_scope = scope
        except Exception:
            continue
    
    # Final validation
    if not _validate(best_rows, n_rows, k):
        best_rows = _pure_greedy(n_rows, k)
    
    return best_rows
