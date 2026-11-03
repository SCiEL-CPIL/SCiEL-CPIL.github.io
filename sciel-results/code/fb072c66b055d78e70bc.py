import time
import random
import sys

def _all_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    L = len(row)
    for i in range(L):
        for j in range(i + 1, L):
            diffs.add(row[j] - row[i])
    return diffs

def _max_entry(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _is_valid(rows):
    """Check that all differences across all rows are distinct."""
    used = set()
    for row in rows:
        row_diffs = _all_diffs(row)
        if used & row_diffs:
            return False
        used |= row_diffs
    return True

def _greedy_build(k, n_rows, rng, max_range=300):
    """
    Build n_rows rows of k+1 increasing integers starting at 0.
    Greedy: for each row, place elements one at a time choosing the smallest
    valid position. Returns list of rows or None on failure.
    """
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for pos in range(1, k + 1):
            # Try candidate positions from last+1 upward
            last = row[-1]
            found = False
            for candidate in range(last + 1, max_range + 1):
                new_diffs = set()
                valid = True
                for existing in row:
                    d = candidate - existing
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(candidate)
                    used_diffs |= new_diffs
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _greedy_build_smart(k, n_rows, rng, max_range=300):
    """
    Greedy with some randomization to escape bad local minima.
    For each element, consider a window of candidates and pick randomly
    from the valid ones (biased toward smaller).
    """
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for pos in range(1, k + 1):
            last = row[-1]
            candidates = []
            for candidate in range(last + 1, min(last + 50, max_range + 1)):
                new_diffs = set()
                valid = True
                for existing in row:
                    d = candidate - existing
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    candidates.append((candidate, new_diffs))
            if not candidates:
                return None
            # Pick from top few with bias toward smaller
            n_pick = min(3, len(candidates))
            chosen = rng.choice(candidates[:n_pick])
            row.append(chosen[0])
            used_diffs |= chosen[1]
        rows.append(row)
    return rows

def _compute_conflicts(rows):
    """
    Return a set of (row_idx, pos) pairs that are 'conflict hotspots' —
    not directly used, but we track which differences are used.
    Returns the global used_diffs set.
    """
    used = set()
    for row in rows:
        used |= _all_diffs(row)
    return used

def _element_move(rows, k, rng):
    """
    Perform a single element move: pick a random row, a random position (not 0),
    and try to move it to a nearby valid position.
    Returns (new_rows, success).
    """
    n = len(rows)
    row_idx = rng.randrange(n)
    row = rows[row_idx][:]
    L = k + 1
    # Pick a position to move (not the first)
    pos = rng.randrange(1, L)
    
    original_val = row[pos]
    
    # Try to find a new valid position
    # Search in a window around current value
    search_radius = rng.choice([3, 5, 10, 20])
    lo = max(row[pos-1] + 1, original_val - search_radius)
    hi = min(row[pos+1] - 1 if pos < L-1 else 1000, original_val + search_radius)
    
    # Collect valid candidates
    candidates = []
    for new_val in range(lo, hi + 1):
        if new_val == original_val:
            continue
        # Check differences with all other elements in this row
        new_row_diffs = set()
        valid = True
        for i in range(L):
            if i == pos:
                continue
            d = abs(new_val - row[i])
            if d in new_row_diffs:
                valid = False
                break
            new_row_diffs.add(d)
        if valid:
            candidates.append(new_val)
    
    if not candidates:
        return None, False
    
    new_val = rng.choice(candidates)
    new_row = row[:]
    new_row[pos] = new_val
    new_rows = rows[:]
    new_rows[row_idx] = new_row
    return new_rows, True

def _try_move_element(rows, k, rng, max_attempts=5):
    """Try to make a beneficial move that reduces scope."""
    current_max = _max_entry(rows)
    for _ in range(max_attempts):
        new_rows, success = _element_move(rows, k, rng)
        if not success:
            continue
        if _is_valid(new_rows):
            new_max = _max_entry(new_rows)
            if new_max <= current_max:
                return new_rows
    return None

def _simulated_annealing(rows, k, deadline, rng, initial_temp=50.0, cooling=0.995):
    """
    Simulated annealing to minimize the maximum entry.
    State: list of rows.
    Cost: max entry (lower is better).
    Move: pick random row, random element, try to move to nearby valid position.
    """
    current = rows[:]
    current_cost = _max_entry(current)
    best = current[:]
    best_cost = current_cost
    temp = initial_temp
    
    while time.time() < deadline:
        # Generate a neighbor
        n = len(current)
        row_idx = rng.randrange(n)
        row = current[row_idx][:]
        L = k + 1
        pos = rng.randrange(1, L)
        
        original_val = row[pos]
        search_radius = rng.choice([2, 4, 8, 15])
        lo = max(row[pos-1] + 1, original_val - search_radius)
        hi = min(row[pos+1] - 1 if pos < L-1 else 200, original_val + search_radius)
        
        candidates = []
        for new_val in range(lo, hi + 1):
            if new_val == original_val:
                continue
            new_row_diffs = set()
            valid = True
            for i in range(L):
                if i == pos:
                    continue
                d = abs(new_val - row[i])
                if d in new_row_diffs:
                    valid = False
                    break
                new_row_diffs.add(d)
            if valid:
                candidates.append(new_val)
        
        if not candidates:
            temp *= cooling
            continue
        
        new_val = rng.choice(candidates)
        new_row = row[:]
        new_row[pos] = new_val
        new_state = current[:]
        new_state[row_idx] = new_row
        
        # Check global validity
        if not _is_valid(new_state):
            temp *= cooling
            continue
        
        new_cost = _max_entry(new_state)
        delta = new_cost - current_cost
        
        if delta <= 0:
            current = new_state
            current_cost = new_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
        else:
            # Accept with probability exp(-delta / temp)
            import math
            if temp > 1e-6:
                prob = math.exp(-delta / temp)
                if rng.random() < prob:
                    current = new_state
                    current_cost = new_cost
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
    
    return best, best_cost

def _rebuild_row_in_context(k, used_diffs_except_row, existing_row, rng, max_val=200):
    """
    Rebuild a single row given that other rows occupy used_diffs_except_row.
    Returns a valid row or None.
    """
    row = [0]
    for pos in range(1, k + 1):
        last = row[-1]
        candidates = []
        for candidate in range(last + 1, max_val + 1):
            new_diffs = set()
            valid = True
            for existing in row:
                d = candidate - existing
                if d in used_diffs_except_row or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                candidates.append(candidate)
        if not candidates:
            return None
        # Pick smallest for determinism, or random from top few
        if rng.random() < 0.7:
            row.append(candidates[0])
        else:
            pick_idx = min(rng.randrange(len(candidates)), 4)
            row.append(candidates[pick_idx])
    return row

def _row_swap_sa(rows, k, deadline, rng):
    """
    SA where a move is to rebuild one entire row given the constraints
    imposed by the other rows. This allows larger structural changes.
    """
    current = rows[:]
    current_cost = _max_entry(current)
    best = current[:]
    best_cost = current_cost
    temp = 30.0
    n = len(current)
    L = k + 1
    diffs_per_row = L * (L - 1) // 2
    
    while time.time() < deadline:
        row_idx = rng.randrange(n)
        
        # Compute used diffs from all OTHER rows
        used_others = set()
        for i in range(n):
            if i != row_idx:
                used_others |= _all_diffs(current[i])
        
        # Rebuild the row
        new_row = _rebuild_row_in_context(k, used_others, current[row_idx], rng, max_val=current_cost + 10)
        if new_row is None:
            temp *= 0.99
            continue
        
        new_state = current[:]
        new_state[row_idx] = new_row
        
        new_cost = _max_entry(new_state)
        delta = new_cost - current_cost
        
        if delta <= 0:
            current = new_state
            current_cost = new_cost
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
        else:
            import math
            if temp > 1e-6:
                prob = math.exp(-delta / temp)
                if rng.random() < prob:
                    current = new_state
                    current_cost = new_cost
        
        temp *= 0.99
        if temp < 0.5:
            temp = 0.5
    
    return best, best_cost

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Phase 1: Get an initial valid solution
    best_solution = None
    best_scope = float('inf')
    
    # Try multiple greedy variants
    for seed in range(10):
        r = random.Random(seed)
        sol = _greedy_build(k, n_rows, r, max_range=400)
        if sol is not None:
            scope = _max_entry(sol)
            if scope < best_scope:
                best_scope = scope
                best_solution = sol[:]
    
    if best_solution is None:
        # Fallback: more random greedy
        for seed in range(20):
            r = random.Random(1000 + seed)
            sol = _greedy_build_smart(k, n_rows, r, max_range=500)
            if sol is not None:
                scope = _max_entry(sol)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = sol[:]
    
    if best_solution is None:
        # Last resort: very generous greedy
        r = random.Random(9999)
        sol = _greedy_build(k, n_rows, r, max_range=1000)
        if sol is not None:
            best_solution = sol
            best_scope = _max_entry(sol)
        else:
            # Should not happen, but return something
            return [[0, 1, 2, 3, 4, 5, 6, 7, 8] for _ in range(n_rows)]
    
    # Phase 2: Element-move local search (quick improvements)
    phase2_deadline = min(start_time + 30.0, deadline - 10.0)
    if time.time() < phase2_deadline:
        improved = True
        while improved and time.time() < phase2_deadline:
            improved = False
            for _ in range(50):
                new_sol = _try_move_element(best_solution, k, rng, max_attempts=1)
                if new_sol is not None:
                    new_scope = _max_entry(new_sol)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = new_sol
                        improved = True
                    break
    
    # Phase 3: Simulated annealing with element moves
    if time.time() < deadline - 5.0:
        sa_deadline = min(start_time + 80.0, deadline - 5.0)
        sa_sol, sa_scope = _simulated_annealing(
            best_solution, k, sa_deadline, rng,
            initial_temp=40.0, cooling=0.997
        )
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_sol
    
    # Phase 4: Row-swap SA (larger structural moves)
    if time.time() < deadline - 3.0:
        rs_deadline = deadline - 3.0
        rs_sol, rs_scope = _row_swap_sa(
            best_solution, k, rs_deadline, rng
        )
        if rs_scope < best_scope:
            best_scope = rs_scope
            best_solution = rs_sol
    
    # Final validation
    if not _is_valid(best_solution):
        # Fallback to a known-good construction
        r = random.Random(42)
        sol = _greedy_build(k, n_rows, r, max_range=500)
        if sol is not None:
            best_solution = sol
    
    return best_solution

if __name__ == "__main__":
    sol = construct(4, 7)
    if sol:
        print("Solution:")
        for row in sol:
            print(" ", row)
        print("Scope:", _max_entry(sol))
        print("Valid:", _is_valid(sol))
    else:
        print("No solution found")
