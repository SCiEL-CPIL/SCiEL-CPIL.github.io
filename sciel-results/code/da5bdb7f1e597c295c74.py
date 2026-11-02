import time
import random
import math

def _scope(rows):
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

def _row_diffs(row):
    """Return set of all pairwise differences in a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diff_set(rows):
    """Return the full set of all differences across all rows."""
    s = set()
    for r in rows:
        s.update(_row_diffs(r))
    return s

def _greedy_construct(n_rows, k, rng, order=None, bias=1.0):
    """
    Greedy construction with randomized element ordering within rows.
    bias > 1 favors smaller values more aggressively.
    """
    if order is None:
        order = list(range(n_rows))
    used = set()
    rows = [None] * n_rows
    
    for row_idx in order:
        row = [0]
        last = 0
        for j in range(1, k + 1):
            # Collect candidates in a window
            candidates = []
            x = last + 1
            max_x = last + 50  # Look ahead up to 50 values
            while x <= max_x and len(candidates) < 8:
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
                x += 1
            
            if not candidates:
                # Fallback: keep searching until we find a valid one
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        candidates.append((x, new_diffs))
                        break
                    x += 1
            
            # Weighted selection: smaller values get higher weight
            weights = [1.0 / (c[0] - last + 1) ** bias for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        
        rows[row_idx] = row
    
    return rows

def _sa_optimize(rows, k, deadline, rng, T0=8.0, cooling=0.9997, min_T=0.01):
    """
    Simulated annealing to minimize scope.
    Moves: pick a random element (not first) and try to change it to a nearby valid value.
    Cost = scope + large penalty for collisions.
    """
    n_rows = len(rows)
    rows = [list(r) for r in rows]
    
    # Precompute per-row diff sets and global diff set
    row_diff_sets = [_row_diffs(r) for r in rows]
    global_diffs = set()
    for ds in row_diff_sets:
        global_diffs.update(ds)
    
    scope = _scope(rows)
    best_scope = scope
    best_rows = [list(r) for r in rows]
    
    T = T0
    penalty = 10000
    
    while time.time() < deadline:
        # Pick random row and element (not the first)
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = rows[i][j]
        prev_val = rows[i][j - 1]
        next_val = rows[i][j + 1] if j < k else scope + 200
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        # Propose new value
        # 70% chance: try to reduce (move down)
        # 30% chance: random perturbation
        if rng.random() < 0.7 and old_val > lo:
            # Move down by a random amount
            max_drop = old_val - lo
            drop = rng.randint(1, max(1, max_drop // 2))
            new_val = old_val - drop
        else:
            span = hi - lo
            if span <= 0:
                continue
            delta = int(rng.gauss(0, max(1, span * 0.15)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue
        
        # Compute old diffs involving position (i, j)
        old_row = rows[i]
        old_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(old_row[t] - old_val)
                old_diffs.add(d)
        
        # Compute new diffs
        new_diffs = set()
        for t in range(k + 1):
            if t != j:
                d = abs(old_row[t] - new_val)
                new_diffs.add(d)
        
        # Check collisions: new_diffs should not intersect with (global_diffs - old_diffs)
        remaining = global_diffs - old_diffs
        collisions = len(new_diffs & remaining)
        
        if collisions > 0:
            # Still might accept at high temperature
            new_scope = max(scope, new_val)
            if new_val == scope and old_val == scope:
                # Need to recompute if scope element changed
                # Approximate: assume scope unchanged unless new_val > scope
                pass
            old_cost = scope + penalty * len(old_diffs & (global_diffs - old_diffs))  # should be 0
            new_cost = new_scope + penalty * collisions
            delta_cost = new_cost - old_cost
        else:
            # No collisions - check if scope improves
            if new_val > scope:
                new_scope = new_val
            elif old_val == scope:
                # Recompute scope
                new_scope = max(r[-1] for r in rows)
                if new_scope == scope:
                    # Check if any other element has value == scope
                    for r in rows:
                        for v in r:
                            if v == scope:
                                new_scope = scope
                                break
                        if new_scope == scope:
                            break
            else:
                new_scope = scope
            
            old_cost = scope
            new_cost = new_scope
            delta_cost = new_cost - old_cost
        
        # Accept if better or with probability exp(-delta/T)
        if delta_cost < 0 or (T > min_T and rng.random() < math.exp(-delta_cost / T)):
            # Accept the move
            global_diffs = remaining | new_diffs
            rows[i][j] = new_val
            scope = new_scope if collisions == 0 else max(scope, new_val)
            
            if collisions == 0 and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                # Occasionally reheat on improvement
                if rng.random() < 0.05:
                    T = min(T * 1.3, 12.0)
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy constructions with different random orders
    best_rows = None
    best_scope = float('inf')
    
    deadline_phase1 = start_time + 20.0
    
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline_phase1:
            break
        
        # Random order of rows
        order = list(range(n_rows))
        rng.shuffle(order)
        
        # Vary bias
        bias = 0.8 + 0.6 * (restart % 5) / 5.0
        
        rows = _greedy_construct(n_rows, k, rng, order=order, bias=bias)
        
        if _valid(rows, k):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Pure greedy fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            for j in range(1, k + 1):
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        last = x
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)
    
    # Phase 2: Simulated annealing optimization
    remaining = deadline - time.time()
    if remaining > 15:
        # Run SA for most of remaining time, leaving buffer
        sa_deadline = time.time() + remaining * 0.85
        sa_scope, sa_rows = _sa_optimize(best_rows, k, sa_deadline, rng)
        
        if _valid(sa_rows, k):
            sa_scope = _scope(sa_rows)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
    elif remaining > 5:
        # Short SA run
        sa_deadline = time.time() + remaining * 0.8
        sa_scope, sa_rows = _sa_optimize(best_rows, k, sa_deadline, rng)
        
        if _valid(sa_rows, k):
            sa_scope = _scope(sa_rows)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
    
    # Final verification and fallback
    if not _valid(best_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            for j in range(1, k + 1):
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        last = x
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
    
    return best_rows
