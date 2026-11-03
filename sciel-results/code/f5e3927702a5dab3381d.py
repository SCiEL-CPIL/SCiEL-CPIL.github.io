import time
import random
import math
from collections import Counter

def _scope(rows):
    if not rows:
        return 0
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

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            s.add(row[j] - row[i])
    return s

def _backtrack_row(pos, current_row, current_diffs, used_diffs, k, max_val, deadline):
    """
    Backtrack to build a single valid row of length k+1 starting at 0,
    avoiding conflicts with used_diffs and internal conflicts.
    Returns the completed row or None.
    """
    if time.time() > deadline:
        return None
    n = k + 1
    if pos == n:
        return list(current_row)
    
    prev = current_row[-1]
    # Try values in a somewhat random order to escape local optima
    # but bias toward smaller values to keep scope small
    min_next = prev + 1
    # We want to keep max_val small, so try values up to max_val
    # but also allow going a bit above if needed
    max_next = max(max_val, prev + k - pos + 1)  # need at least this many more steps
    
    # Generate candidate values, try smaller ones first with some randomness
    candidates = list(range(min_next, max_next + 1))
    # Shuffle the upper half to add diversity
    if len(candidates) > 4:
        mid = len(candidates) // 2
        random.shuffle(candidates[mid:])
    
    for val in candidates:
        if time.time() > deadline:
            return None
        # Check conflicts
        conflict = False
        new_diffs = []
        for i in range(pos):
            d = val - current_row[i]
            if d in current_diffs or d in used_diffs:
                conflict = True
                break
            new_diffs.append(d)
        
        if conflict:
            continue
        
        current_row.append(val)
        current_diffs.update(new_diffs)
        
        result = _backtrack_row(pos + 1, current_row, current_diffs, used_diffs, k, max_val, deadline)
        
        current_row.pop()
        for d in new_diffs:
            current_diffs.discard(d)
        
        if result is not None:
            return result
    
    return None

def _construct_rows(n_rows, k, deadline, rng, target_scope=None):
    """
    Construct n_rows valid rows using backtracking with randomization.
    Returns list of rows or None.
    """
    if target_scope is None:
        # Estimate a reasonable target scope
        # Each row has C(k+1, 2) differences
        total_diffs = n_rows * (k + 1) * k // 2
        # Differences range from 1 to scope, so we need scope >= total_diffs roughly
        # But differences aren't all distinct values from 1..scope, they're specific
        # A loose bound: scope should be at least total_diffs / 2 or so
        target_scope = max(105, total_diffs)
    
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        # Try multiple times with different random seeds/orderings
        best_row = None
        best_max_val = float('inf')
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Start backtracking
            current_row = [0]
            current_diffs = set()
            
            result = _backtrack_row(1, current_row, current_diffs, used_diffs, k, target_scope, deadline)
            
            if result is not None:
                row_max = result[-1]
                if row_max < best_max_val:
                    best_max_val = row_max
                    best_row = result
            
            # If we found a row with reasonable scope, accept it
            if best_row is not None and best_max_val <= target_scope:
                break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        for d in _diffs_for_row(best_row):
            used_diffs.add(d)
    
    return rows

def _sa_optimize(rows, k, deadline, rng, max_iters=100000):
    """
    Simulated Annealing to minimize the scope with collision penalty.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    # Track all differences for quick collision checking
    all_diffs = []
    for r in current:
        all_diffs.extend(_diffs_for_row(r))
    diff_count = Counter(all_diffs)
    collisions = sum(max(0, c - 1) for c in diff_count.values())
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    T = 100.0
    cooling = 0.9995
    min_T = 0.1
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Choose a random row and position to modify
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 1000
        
        # Determine move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Scope compression: try to reduce a value, especially if it's near the max
            if old_val == current_scope:
                # Strongly bias toward reducing the max
                new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val - 1))
                if new_val >= old_val:
                    new_val = old_val - 1
            else:
                new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val))
            if new_val <= prev_val:
                continue
        elif move_type < 0.8:
            # Small perturbation
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
        else:
            # Larger perturbation
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 20))
        
        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        
        # Compute the difference changes
        old_diffs = _diffs_for_row(current[i])
        old_diffs_set = set(old_diffs)
        
        # Update the row temporarily
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)
        new_diffs_set = set(new_diffs)
        
        # Check validity of new row
        valid_row = len(new_diffs) == len(new_diffs_set) and all(d > 0 for d in new_diffs)
        
        if not valid_row:
            current[i][j] = old_val
            continue
        
        # Compute scope change
        new_scope = max(current_scope, max(new_row))
        # If we reduced the max, recompute scope
        if old_val == current_scope and new_val < old_val:
            new_scope = max(max(r) for r in current)
        
        # Compute collision change efficiently
        # Remove old diffs, add new diffs
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        
        new_collisions = sum(max(0, c - 1) for c in diff_count.values())
        
        # Cost function: heavily penalize collisions, then scope
        collision_penalty = 10000
        old_cost = collisions * collision_penalty + current_scope
        new_cost = new_collisions * collision_penalty + new_scope
        
        # Accept if better or with probability based on temperature
        if new_cost <= old_cost:
            accept = True
        else:
            delta_cost = new_cost - old_cost
            accept_prob = math.exp(-delta_cost / max(T, min_T))
            accept = rng.random() < accept_prob
        
        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def _scope_squeeze(rows, k, deadline, rng):
    """
    Try to reduce the scope by 1 iteratively.
    """
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    
    # Find which row(s) have the max value
    max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
    
    # Try to reduce the max element in those rows
    for _ in range(50):  # Try up to 50 times
        if time.time() > deadline:
            break
        
        if not max_rows:
            break
        
        # Pick a random row with max value
        i = rng.choice(max_rows)
        row = rows[i]
        old_max = row[-1]
        
        # Try to reduce the last element
        # The last element is row[k], and it must be > row[k-1]
        prev = row[k-1]
        
        # Try values from old_max-1 down to prev+1
        for new_val in range(old_max - 1, prev, -1):
            # Check if this creates a conflict
            # Remove old diffs from row, add new diffs
            old_diffs = _diffs_for_row(row)
            old_diffs_set = set(old_diffs)
            
            row[k] = new_val
            new_diffs = _diffs_for_row(row)
            new_diffs_set = set(new_diffs)
            
            # Check internal validity
            if len(new_diffs) != len(new_diffs_set):
                row[k] = old_max
                continue
            
            # Check cross-row conflicts
            # We need to check if any new diff conflicts with diffs in other rows
            # This is expensive, so let's do a quick check
            conflict = False
            for other_idx, other_row in enumerate(rows):
                if other_idx == i:
                    continue
                other_diffs = _diffs_for_row(other_row)
                for d in new_diffs_set:
                    if d in other_diffs:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                # Accept the change
                current_scope = new_val
                max_rows = [idx for idx, r in enumerate(rows) if r[-1] == current_scope]
                break
            else:
                row[k] = old_max
        
        if not max_rows:
            break
    
    return rows

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin
    rng = random.Random(42)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Try to construct rows using backtracking
    # Start with a reasonable target scope and try to find a valid configuration
    for target_scope in [112, 115, 120, 130, 140]:
        if time.time() > deadline - 20:
            break
        
        rows = _construct_rows(n_rows, k, deadline - 15, rng, target_scope)
        
        if rows is not None:
            s = _scope(rows)
            if s < best_score:
                best_score = s
                best_result = rows
            
            # If we found a solution within target scope, we can stop
            if s <= target_scope:
                break
    
    # Phase 2: If we didn't find anything, use a simple greedy fallback
    if best_result is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_result = rows
        best_score = _scope(rows)
    
    # Phase 3: Simulated Annealing Polish
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining * 0.7
        s, rows = _sa_optimize(best_result, k, sa_deadline, rng, max_iters=int(remaining * 30000))
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 4: Scope Squeeze
    if time.time() < deadline - 5:
        remaining = deadline - time.time()
        squeeze_deadline = time.time() + remaining * 0.8
        squeezed = _scope_squeeze(best_result, k, squeeze_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed
    
    return best_result
