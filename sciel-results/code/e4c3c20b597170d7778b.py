"""Difference triangle set (7,5): minimize the largest entry.
Structural change: model as CSP over (row, position) pairs with
constraint propagation on difference conflicts. Use multi-start
random restarts with different row orderings and value heuristics.
"""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    n = len(row)
    diffs = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ri)
    return diffs


def _all_diffs(rows):
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _valid(rows):
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, seed=42):
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    break
                x += 1
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows


def _csp_backtrack(n_rows, k, max_val, rng, deadline, row_order=None, val_heuristic='random'):
    """
    CSP backtracking: assign positions (row_idx, pos_idx) in a given order.
    Each variable: the value at that position.
    Domain: integers strictly between neighbors (or 0..max_val for first).
    Constraint: all differences must be globally unique.
    
    Uses forward checking: after assigning a value, check that remaining
    variables still have non-empty domains (at least one value that doesn't
    conflict).
    """
    if row_order is None:
        row_order = list(range(n_rows))
    
    # Variables: (row_idx, pos_idx) for pos_idx in 1..k
    # We'll assign in the order: for each row in row_order, positions 1..k
    # But we can interleave rows for better constraint propagation
    
    # Build assignment order: interleave positions across rows
    # Strategy: assign position p across all rows before moving to p+1
    # This maximizes constraint propagation between rows at the same "level"
    
    # Actually, let's try: assign all of row 0, then row 1, etc.
    # But with random row ordering.
    
    # Alternative: assign by "depth" - position 1 for all rows, then position 2, etc.
    # This creates more cross-row constraints early.
    
    # Let's use: for pos in 1..k: for row in row_order: assign (row, pos)
    # This means we fix the "shape" level by level.
    
    used = set()
    # rows[r][p] = value, with rows[r][0] = 0
    rows = [[0] * (k + 1) for _ in range(n_rows)]
    
    # Precompute: for each (row, pos), what are the constraints from already-assigned values?
    # We track: for each row, the assigned values so far.
    
    # Assignment order
    order = []
    for pos in range(1, k + 1):
        for r in row_order:
            order.append((r, pos))
    
    def get_domain(r, pos, rows, used):
        """Get valid values for position (r, pos)."""
        row = rows[r]
        # Must be > row[pos-1] and < row[pos+1] (if assigned)
        low = row[pos - 1] + 1
        high = max_val
        if pos + 1 <= k and row[pos + 1] > 0:  # already assigned
            high = min(high, row[pos + 1] - 1)
        
        # Also need: for all already-assigned positions in this row,
        # the differences must not be in used
        assigned_in_row = [p for p in range(k + 1) if p != pos and row[p] > 0 or p == 0]
        # Actually row[0] = 0 always, and others are 0 if unassigned
        assigned_in_row = [p for p in range(k + 1) if p == 0 or (1 <= p <= k and row[p] > 0)]
        
        valid = []
        for v in range(low, high + 1):
            # Check differences with all assigned positions in this row
            conflict = False
            for p in assigned_in_row:
                if p == pos:
                    continue
                d = abs(v - row[p])
                if d == 0:
                    continue
                if d in used:
                    conflict = True
                    break
            if not conflict:
                valid.append(v)
        return valid
    
    def forward_check(r, pos, val, rows, used, order, current_idx):
        """After assigning (r, pos) = val, check if all remaining vars have non-empty domain."""
        new_used = set(used)
        row = rows[r]
        # Add new differences
        for p in range(k + 1):
            if p == pos:
                continue
            if p == 0 or (1 <= p <= k and row[p] > 0):
                d = abs(val - row[p])
                if d > 0:
                    new_used.add(d)
        
        # Check remaining variables
        for idx in range(current_idx + 1, len(order)):
            r2, pos2 = order[idx]
            row2 = rows[r2]
            low2 = row2[pos2 - 1] + 1
            high2 = max_val
            if pos2 + 1 <= k and row2[pos2 + 1] > 0:
                high2 = min(high2, row2[pos2 + 1] - 1)
            
            if low2 > high2:
                return False
            
            # Quick check: is there at least one value in [low2, high2] that doesn't conflict?
            # This is expensive; do a sampled check for speed
            # Check a few candidate values
            assigned_in_row2 = [p for p in range(k + 1) if p == 0 or (1 <= p <= k and row2[p] > 0)]
            
            # Sample up to 5 values
            step = max(1, (high2 - low2 + 1) // 5)
            found = False
            for v in range(low2, high2 + 1, step):
                conflict = False
                for p in assigned_in_row2:
                    if p == pos2:
                        continue
                    d = abs(v - row2[p])
                    if d == 0:
                        continue
                    if d in new_used:
                        conflict = True
                        break
                if not conflict:
                    found = True
                    break
            if not found:
                # Try a couple more
                for v in [low2, high2, (low2 + high2) // 2]:
                    if v < low2 or v > high2:
                        continue
                    conflict = False
                    for p in assigned_in_row2:
                        if p == pos2:
                            continue
                        d = abs(v - row2[p])
                        if d == 0:
                            continue
                        if d in new_used:
                            conflict = True
                            break
                    if not conflict:
                        found = True
                        break
            if not found:
                return False
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == len(order):
            return [list(rows[r]) for r in range(n_rows)]
        
        r, pos = order[idx]
        row = rows[r]
        low = row[pos - 1] + 1
        high = max_val
        if pos + 1 <= k and row[pos + 1] > 0:
            high = min(high, row[pos + 1] - 1)
        
        if low > high:
            return None
        
        # Get candidates
        candidates = list(range(low, high + 1))
        
        if val_heuristic == 'random':
            rng.shuffle(candidates)
        elif val_heuristic == 'small_first':
            pass  # already in order
        elif val_heuristic == 'large_first':
            candidates.reverse()
        elif val_heuristic == 'middle':
            mid = (low + high) // 2
            candidates.sort(key=lambda v: abs(v - mid))
        
        # Filter candidates by immediate constraint
        assigned_in_row = [p for p in range(k + 1) if p == 0 or (1 <= p <= k and row[p] > 0)]
        
        filtered = []
        for v in candidates:
            conflict = False
            for p in assigned_in_row:
                if p == pos:
                    continue
                d = abs(v - row[p])
                if d == 0:
                    continue
                if d in used:
                    conflict = True
                    break
            if not conflict:
                filtered.append(v)
        
        if not filtered:
            return None
        
        for v in filtered:
            if time.time() > deadline:
                return None
            
            # Assign
            row[pos] = v
            new_diffs = set()
            for p in assigned_in_row:
                if p == pos:
                    continue
                d = abs(v - row[p])
                if d > 0:
                    new_diffs.add(d)
            used.update(new_diffs)
            
            # Forward check
            if forward_check(r, pos, v, rows, used, order, idx):
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            
            # Undo
            used.difference_update(new_diffs)
            row[pos] = 0
        
        return None
    
    result = backtrack(0)
    return result


def _multi_start_csp(n_rows, k, max_val, rng, deadline, num_starts=5):
    """Try multiple CSP restarts with different row orderings and heuristics."""
    best = None
    best_scope = max_val + 1
    
    heuristics = ['random', 'small_first', 'large_first', 'middle']
    
    for start in range(num_starts):
        if time.time() > deadline:
            break
        
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        h = heuristics[start % len(heuristics)]
        
        result = _csp_backtrack(n_rows, k, max_val, rng, deadline, row_order=row_order, val_heuristic=h)
        if result is not None:
            s = _scope(result)
            if s < best_scope:
                best_scope = s
                best = result
    
    return best, best_scope


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce the maximum entry by modifying individual elements."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope:
                    continue
                
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                candidates = list(range(high, low - 1, -1))
                rng.shuffle(candidates)
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
                        continue
                    
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(old_val - row[i])
                    for i in range(pos + 1, len(row)):
                        old_diffs.add(row[i] - old_val)
                    
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for i in range(pos + 1, len(row)):
                        new_diffs.add(row[i] - new_val)
                    
                    conflict = new_diffs & (used - old_diffs)
                    if conflict:
                        continue
                    
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    improved = True
                    break
    
    return rows


def _difference_swap(rows, k, deadline, rng):
    """
    Try swapping two elements within a row (if they maintain increasing order)
    or moving an element to a different position, to free up difference conflicts.
    """
    rows = [list(r) for r in rows]
    used = _all_diffs(rows)
    best = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    steps = 0
    while time.time() < deadline and steps < 5000:
        steps += 1
        ri = rng.randint(0, len(rows) - 1)
        row = rows[ri]
        
        # Pick two positions to try to modify
        p1 = rng.randint(1, len(row) - 1)
        
        old_val = row[p1]
        low = row[p1 - 1] + 1
        high = row[p1 + 1] - 1 if p1 + 1 < len(row) else old_val + 5
        
        if low >= high:
            continue
        
        new_val = rng.randint(low, high)
        if new_val == old_val:
            continue
        
        old_diffs = set()
        for i in range(p1):
            old_diffs.add(old_val - row[i])
        for i in range(p1 + 1, len(row)):
            old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(p1):
            new_diffs.add(new_val - row[i])
        for i in range(p1 + 1, len(row)):
            new_diffs.add(row[i] - new_val)
        
        conflict = new_diffs & (used - old_diffs)
        if not conflict:
            row[p1] = new_val
            used -= old_diffs
            used |= new_diffs
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best = [list(r) for r in rows]
    
    return best


def _polish_pass(rows, k, used, deadline, rng):
    """Final polish: try to reduce max by 1."""
    rows = [list(r) for r in rows]
    used = set(used)
    
    current_scope = _scope(rows)
    target = current_scope - 1
    
    while target >= 100 and time.time() < deadline:
        success = False
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            trial_rows = [list(r) for r in rows]
            trial_used = set(used)
            
            row_changed = False
            for ri in range(len(trial_rows)):
                row = trial_rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    old_val = row[pos]
                    if old_val <= target:
                        continue
                    
                    low = row[pos - 1] + 1
                    high = min(target, row[pos + 1] - 1 if pos + 1 < len(row) else target)
                    
                    if low > high:
                        continue
                    
                    for new_val in range(high, low - 1, -1):
                        if new_val >= old_val:
                            continue
                        
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for i in range(pos + 1, len(row)):
                            old_diffs.add(row[i] - old_val)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        
                        conflict = new_diffs & (trial_used - old_diffs)
                        if conflict:
                            continue
                        
                        row[pos] = new_val
                        trial_used -= old_diffs
                        trial_used |= new_diffs
                        row_changed = True
                        break
                    if row_changed and row[pos] <= target:
                        break
            
            if row_changed and _scope(trial_rows) <= target:
                if _valid(trial_rows):
                    rows = trial_rows
                    used = trial_used
                    success = True
                    break
        
        if success:
            current_scope = _scope(rows)
            target = current_scope - 1
        else:
            break
    
    return rows, used


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Quick greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: CSP backtracking with multi-start
    # Try decreasing max_val bounds
    if time.time() < deadline - 30.0:
        # Start from greedy scope, try to improve
        for max_val in range(best_scope, max(104, best_scope - 15), -1):
            if time.time() > deadline - 30.0:
                break
            
            # Try a few restarts
            num_starts = 3 if (best_scope - max_val) < 5 else 2
            result, scope = _multi_start_csp(n_rows, k, max_val, rng, deadline - 30.0, num_starts=num_starts)
            if result is not None:
                if scope < best_scope:
                    best_rows = result
                    best_scope = scope
                    break
    
    # Phase 3: Local search
    if time.time() < deadline - 20.0:
        improved = _local_search_improve(best_rows, k, deadline - 20.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 4: Difference swap local search
    if time.time() < deadline - 10.0:
        swapped = _difference_swap(best_rows, k, deadline - 10.0, rng)
        new_scope = _scope(swapped)
        if new_scope < best_scope:
            best_rows = swapped
            best_scope = new_scope
    
    # Phase 5: Another round of CSP with tighter bound
    if time.time() < deadline - 15.0 and best_scope > 105:
        target = best_scope - 1
        result, scope = _multi_start_csp(n_rows, k, target, rng, deadline - 15.0, num_starts=3)
        if result is not None and scope < best_scope:
            best_rows = result
            best_scope = scope
    
    # Phase 6: Final polish
    if time.time() < deadline - 2.0:
        used = _all_diffs(best_rows)
        polished, used = _polish_pass(best_rows, k, used, deadline - 2.0, rng)
        new_scope = _scope(polished)
        if new_scope < best_scope:
            best_rows = polished
            best_scope = new_scope
    
    # Verify validity
    if not _valid(best_rows):
        # Fallback to greedy
        best_rows = _greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
