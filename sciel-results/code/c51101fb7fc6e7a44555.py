"""Difference Triangle Set (4,7) solver.

Structural change: use a difference-set-centric construction where we
think in terms of which differences are "available" and build rows by
selecting the smallest feasible value that doesn't create a conflict.

Key structural improvements:
1. Difference-aware greedy: instead of incrementing x by 1 and checking,
   we maintain a sorted list of "forbidden" differences and pick the
   smallest x that avoids all of them.
2. Row-permutation search: try all 4! = 24 row orderings with different
   greedy strategies, since the order in which rows are built affects
   the difference pool.
3. Iterated local search with difference-based repair: when stuck,
   randomly remove a row's differences, rebuild that row using the
   difference-aware method, and recompact.
4. Time-bounded with explicit deadline guard.
"""
import time
import random
import math


def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _build_used_set(rows, k):
    L = k + 1
    used = set()
    for row in rows:
        for i in range(L):
            for j in range(i + 1, L):
                used.add(row[j] - row[i])
    return used


def _row_diffs(row):
    """Return set of all positive differences in a row."""
    L = len(row)
    diffs = set()
    for i in range(L):
        for j in range(i + 1, L):
            diffs.add(row[j] - row[i])
    return diffs


def _diff_aware_greedy(n_rows, k, rng, used_set, max_bound=None, row_order=None, 
                       jump_strategy='uniform', jump_max=3, skip_p=0.0):
    """Build rows using difference-aware selection.
    
    For each row, instead of incrementing x, we find the smallest x > prev
    such that all new differences (x - a for a in current_row) are not in used_set.
    """
    if row_order is None:
        row_order = list(range(n_rows))
    
    rows = [None] * n_rows
    L = k + 1
    
    for r_idx in row_order:
        row = [0]
        used_local = set()
        prev = 0
        
        while len(row) < L:
            # Find smallest x > prev such that {x - a for a in row} doesn't intersect used_set
            found = False
            x = prev + 1
            
            # We can be smarter: the forbidden x values are those where
            # x - a in used_set for some a in row, i.e., x in {a + d for a in row, d in used_set}
            # But that's expensive. Instead, just scan forward.
            # Optimization: skip ahead by checking if any a + d == x for d in used_set
            
            while not found:
                if max_bound is not None and x > max_bound:
                    return None
                
                # Check if x creates any conflict
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_set or d in used_local:
                        conflict = True
                        break
                
                if not conflict:
                    # Also check that x - a values are all distinct among themselves
                    # (they are, since a values are distinct and x is fixed)
                    found = True
                    row.append(x)
                    used_local.update(x - a for a in row[:-1])
                    prev = x
                else:
                    x += 1
                    
                    # Optional: skip more aggressively
                    if skip_p > 0 and rng.random() < skip_p:
                        x += rng.randint(1, jump_max)
                    
                    # Jump strategy for diversity
                    if jump_strategy == 'random' and rng.random() < 0.1:
                        x += rng.randint(1, jump_max)
        
        if max_bound is not None and row[-1] > max_bound:
            return None
        
        rows[r_idx] = row
        used_set.update(used_local)
    
    return rows


def _greedy_construct(n_rows, k, rng=None):
    """Simple greedy baseline."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _compaction_pass(rows, n_rows, k, time_limit):
    """Iteratively compact each element to its smallest feasible value."""
    work = [list(r) for r in rows]
    used_set = _build_used_set(work, k)
    L = k + 1
    improved = True
    
    while improved and time.time() < time_limit:
        improved = False
        for r_idx in range(n_rows):
            row = work[r_idx]
            # Process right to left
            for pos in range(L - 1, 0, -1):
                old_val = row[pos]
                lower = row[pos - 1] + 1
                if lower >= old_val:
                    continue
                
                # Get current diffs involving this position
                old_diffs = set()
                for i in range(L):
                    if i < pos:
                        old_diffs.add(old_val - row[i])
                    elif i > pos:
                        old_diffs.add(row[i] - old_val)
                
                # Try to decrease
                for new_val in range(lower, old_val):
                    if pos < L - 1 and new_val >= row[pos + 1]:
                        break
                    
                    new_diffs = set()
                    for i in range(L):
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        elif i > pos:
                            new_diffs.add(row[i] - new_val)
                    
                    other = used_set - old_diffs
                    if new_diffs & other:
                        continue
                    
                    # Apply
                    for d in old_diffs:
                        used_set.discard(d)
                    for d in new_diffs:
                        used_set.add(d)
                    row[pos] = new_val
                    improved = True
                    break
    
    return work


def _row_rebuild(r_idx, work, used_set, L, rng, max_val):
    """Rebuild a single row using difference-aware greedy."""
    row = work[r_idx]
    old_diffs = _row_diffs(row)
    for d in old_diffs:
        used_set.discard(d)
    
    best_new_row = None
    best_new_max = max_val + 1
    
    for attempt in range(30):
        new_row = [0]
        used_local = set()
        prev = 0
        valid = True
        
        while len(new_row) < L:
            x = prev + 1
            found = False
            
            while not found:
                if x > best_new_max:
                    valid = False
                    break
                
                conflict = False
                for a in new_row:
                    d = x - a
                    if d in used_set or d in used_local:
                        conflict = True
                        break
                
                if not conflict:
                    found = True
                    new_row.append(x)
                    used_local.update(x - a for a in new_row[:-1])
                    prev = x
                else:
                    x += 1
                    if rng.random() < 0.05:
                        x += rng.randint(1, 3)
            
            if not valid:
                break
        
        if valid and len(new_row) == L:
            new_max = new_row[-1]
            if new_max < best_new_max:
                best_new_max = new_max
                best_new_row = list(new_row)
    
    if best_new_row is not None:
        work[r_idx] = best_new_row
        used_set.update(_row_diffs(best_new_row))
        return True
    else:
        work[r_idx] = row
        used_set.update(old_diffs)
        return False


def _iterated_local_search(rows, n_rows, k, rng, time_limit):
    """Iterated local search with row rebuilds and compaction."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = _build_used_set(work, k)
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max
    
    stagnation = 0
    max_stagnation = 500
    
    while time.time() < time_limit:
        move_type = rng.random()
        
        if move_type < 0.4:
            # Rebuild a random row
            r_idx = rng.randint(0, n_rows - 1)
            _row_rebuild(r_idx, work, used_set, L, rng, current_max)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation = 0
            else:
                stagnation += 1
                
        elif move_type < 0.7:
            # Compaction pass
            work = _compaction_pass(work, n_rows, k, time.time() + 0.5)
            used_set = _build_used_set(work, k)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation = 0
            else:
                stagnation += 1
                
        elif move_type < 0.85:
            # Perturb: increase a random element, then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            
            # Get diffs involving this position
            old_diffs = set()
            for i in range(L):
                if i < pos:
                    old_diffs.add(old_val - row[i])
                elif i > pos:
                    old_diffs.add(row[i] - old_val)
            
            # Try to increase
            delta = rng.randint(1, max(2, current_max // 4))
            new_val = old_val + delta
            
            if pos < L - 1 and new_val >= row[pos + 1]:
                stagnation += 1
            else:
                new_diffs = set()
                for i in range(L):
                    if i < pos:
                        new_diffs.add(new_val - row[i])
                    elif i > pos:
                        new_diffs.add(row[i] - new_val)
                
                other = used_set - old_diffs
                if not (new_diffs & other):
                    for d in old_diffs:
                        used_set.discard(d)
                    for d in new_diffs:
                        used_set.add(d)
                    row[pos] = new_val
                    
                    # Recompress this row
                    for p in range(pos + 1, L):
                        row[p] = _compaction_pass([row], 1, k, time.time() + 0.1)[0][p]
                    
                    new_max = _max_entry(work)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in work]
                        current_max = best_max
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1
        else:
            # Full restart with new random ordering
            order = list(range(n_rows))
            rng.shuffle(order)
            fresh_used = set()
            fresh_rows = _diff_aware_greedy(n_rows, k, rng, fresh_used, 
                                            max_bound=current_max, row_order=order)
            if fresh_rows is not None and _check_valid(fresh_rows, k):
                fresh_max = _max_entry(fresh_rows)
                if fresh_max < best_max:
                    best_max = fresh_max
                    best_rows = [list(r) for r in fresh_rows]
                    work = [list(r) for r in fresh_rows]
                    used_set = _build_used_set(work, k)
                    current_max = best_max
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1
        
        if stagnation > max_stagnation:
            # Restart with fresh greedy
            fresh_used = set()
            order = list(range(n_rows))
            rng.shuffle(order)
            fresh_rows = _diff_aware_greedy(n_rows, k, rng, fresh_used,
                                            row_order=order)
            if fresh_rows is not None and _check_valid(fresh_rows, k):
                fresh_max = _max_entry(fresh_rows)
                if fresh_max < best_max:
                    best_max = fresh_max
                    best_rows = [list(r) for r in fresh_rows]
                    work = [list(r) for r in fresh_rows]
                    used_set = _build_used_set(work, k)
                    current_max = best_max
            stagnation = 0
    
    return best_rows


def _mrv_backtrack(n_rows, k, rng, time_limit, max_bound=None):
    """MRV-based backtracking CSP solver."""
    L = k + 1
    total_vars = n_rows * k
    
    best_solution = None
    best_max = max_bound if max_bound else float('inf')
    start_time = time.time()
    
    def backtrack(depth, rows, used_diffs, assigned):
        nonlocal best_solution, best_max, start_time
        
        if time.time() - start_time > time_limit:
            return False
        
        if depth == total_vars:
            row_max = max(max(r) for r in rows)
            if row_max < best_max:
                best_max = row_max
                best_solution = [list(r) for r in rows]
            return True
        
        # MRV: find unassigned variable with minimum domain size
        min_size = float('inf')
        best_r, best_pos = -1, -1
        
        for r in range(n_rows):
            for pos in range(1, L):
                if assigned[r][pos]:
                    continue
                
                lower = rows[r][pos-1] + 1
                upper = rows[r][pos+1] - 1 if pos < L-1 else (best_max - 1 if best_max < float('inf') else 200)
                
                if lower >= upper:
                    return False
                
                count = 0
                for val in range(lower, min(upper + 1, best_max if best_max < float('inf') else 200)):
                    ok = True
                    for i in range(L):
                        if i == pos or not assigned[r][i]:
                            continue
                        d = abs(val - rows[r][i])
                        if d in used_diffs:
                            ok = False
                            break
                    if ok:
                        count += 1
                        if count >= min_size:
                            break
                
                if count == 0:
                    return False
                if count < min_size:
                    min_size = count
                    best_r, best_pos = r, pos
                    if count == 1:
                        break
            if min_size == 1:
                break
        
        if best_r == -1:
            return False
        
        # Get domain
        domain = []
        lower = rows[best_r][best_pos-1] + 1
        upper = rows[best_r][best_pos+1] - 1 if best_pos < L-1 else (best_max - 1 if best_max < float('inf') else 200)
        
        for val in range(lower, min(upper + 1, best_max if best_max < float('inf') else 200)):
            ok = True
            for i in range(L):
                if i == best_pos or not assigned[best_r][i]:
                    continue
                d = abs(val - rows[best_r][i])
                if d in used_diffs:
                    ok = False
                    break
            if ok:
                domain.append(val)
        
        if not domain:
            return False
        
        rng.shuffle(domain)
        
        for val in domain:
            if time.time() - start_time > time_limit:
                return False
            
            rows[best_r][best_pos] = val
            assigned[best_r][best_pos] = True
            
            new_diffs = set()
            for i in range(L):
                if i != best_pos and assigned[best_r][i]:
                    d = abs(val - rows[best_r][i])
                    new_diffs.add(d)
            
            conflict = any(d in used_diffs for d in new_diffs)
            
            if not conflict:
                used_diffs.update(new_diffs)
                
                if backtrack(depth + 1, rows, used_diffs, assigned):
                    return True
                
                used_diffs.difference_update(new_diffs)
            
            assigned[best_r][best_pos] = False
            rows[best_r][best_pos] = 0
        
        return False
    
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        rows = [[0] * L for _ in range(n_rows)]
        used_diffs = set()
        assigned = [[False] * L for _ in range(n_rows)]
        for r in range(n_rows):
            assigned[r][0] = True
        
        if backtrack(0, rows, used_diffs, assigned):
            if best_solution is not None:
                return best_solution
    
    return best_solution


def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 140  # Safety margin before 150s kill

    rng = random.Random(42)
    
    # 1) Baseline greedy
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)
    
    # 2) Difference-aware greedy with various row orderings and parameters
    greedy_deadline = time_limit - 60
    for trial in range(200):
        if time.time() >= greedy_deadline:
            break
        
        order = list(range(n_rows))
        rng.shuffle(order)
        
        used_set = set()
        jump_strategy = rng.choice(['uniform', 'random'])
        jump_max = rng.randint(1, 5)
        skip_p = rng.random() * 0.1
        
        rows = _diff_aware_greedy(n_rows, k, rng, used_set,
                                  row_order=order,
                                  jump_strategy=jump_strategy,
                                  jump_max=jump_max,
                                  skip_p=skip_p)
        
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax
    
    # 3) MRV backtracking with current best as bound
    mrv_deadline = time_limit - 40
    if time.time() < mrv_deadline:
        mrv_result = _mrv_backtrack(n_rows, k, rng, mrv_deadline - time.time(), 
                                    max_bound=best_max)
        if mrv_result is not None and _check_valid(mrv_result, k):
            mmax = _max_entry(mrv_result)
            if mmax < best_max:
                best_rows = mrv_result
                best_max = mmax
    
    # 4) Compaction
    compacted = _compaction_pass(best_rows, n_rows, k, time_limit - 20)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax
    
    # 5) Iterated local search with remaining time
    if time.time() < time_limit - 3:
        ils_result = _iterated_local_search(best_rows, n_rows, k, rng, time_limit - 3)
        if _check_valid(ils_result, k):
            imax = _max_entry(ils_result)
            if imax < best_max:
                best_rows = ils_result
                best_max = imax
    
    # 6) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    
    return best_rows
