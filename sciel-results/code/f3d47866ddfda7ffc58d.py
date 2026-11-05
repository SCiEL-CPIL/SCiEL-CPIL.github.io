"""Difference triangle set (7,5): minimize the largest entry.
Structural redesign: Use set-based difference tracking to correctly handle
duplicate differences within rows (Sidon property per row) and across rows.
Replaces bitset logic which failed to detect internal duplicates."""
import time
import random

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        
        # Check for internal duplicates first
        rd = _row_diffs_set(row)
        num_pairs = len(row) * (len(row) - 1) // 2
        if len(rd) != num_pairs:
            return False
        
        # Check for global duplicates
        if rd & all_d:
            return False
        all_d |= rd
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization.
    Now correctly checks for internal duplicates."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        local_diffs = set()
        while len(row) < k + 1:
            found = False
            # Try random steps first if randomize
            if randomize:
                for _attempt in range(20):
                    step = rng.randint(1, 50)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    # Check internal and global
                    if not (new_diffs & local_diffs) and not (new_diffs & used):
                        found = True
                        break
            
            if not found:
                # Fallback to linear scan
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_diffs) and not (new_diffs & used):
                        break
                    x += 1
            
            row.append(x)
            local_diffs |= new_diffs
            used |= new_diffs
        rows.append(row)
    
    return rows

def _backtrack_row(k, used_global, max_val, rng, deadline=None):
    """Try to build a single row of k+1 elements starting at 0.
    Ensures no internal duplicates and no global conflicts."""
    if deadline is not None and time.time() > deadline:
        return None
    
    row = [0]
    local_diffs = set()
    
    def backtrack(idx):
        if deadline is not None and time.time() > deadline:
            return False
        if idx == k + 1:
            return True
        
        low = row[-1] + 1
        high = max_val
        
        # Heuristic: try smaller steps first to keep scope low? Or random?
        # Randomization helps escape local minima.
        candidates = list(range(low, high + 1))
        rng.shuffle(candidates)
        
        for val in candidates:
            new_diffs = {val - a for a in row}
            # Check internal duplicates
            if new_diffs & local_diffs:
                continue
            # Check global duplicates
            if new_diffs & used_global:
                continue
            
            row.append(val)
            local_diffs.update(new_diffs)
            
            if backtrack(idx + 1):
                return True
            
            row.pop()
            local_diffs.difference_update(new_diffs)
        
        return False
    
    if backtrack(1):
        return row
    return None

def _construct_with_backtracking(n_rows, k, max_val, rng, deadline):
    """Construct n_rows rows using backtracking."""
    used_global = set()
    rows = []
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = _backtrack_row(k, used_global, max_val, rng, deadline)
        if row is None:
            return None
        rows.append(row)
        used_global |= _row_diffs_set(row)
    
    return rows

def _local_search_improve(rows, k, deadline, rng=None):
    """Local search to reduce scope.
    Correctly handles set-based difference updates."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute global used set
    used_global = set()
    for r in rows:
        used_global |= _row_diffs_set(r)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            # Compute current diffs for this row
            row_diffs = _row_diffs_set(row)
            
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope:
                    continue
                
                # Try to reduce value
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                # Try values
                candidates = list(range(high, low - 1, -1))
                rng.shuffle(candidates)
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
                        continue
                    
                    # Calculate new diffs for this row
                    new_row_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            if i == pos or j == pos:
                                if i == pos:
                                    new_row_diffs.add(row[j] - new_val)
                                elif j == pos:
                                    new_row_diffs.add(new_val - row[i])
                            else:
                                new_row_diffs.add(row[j] - row[i])
                    
                    # Check internal duplicates
                    num_pairs = len(row) * (len(row) - 1) // 2
                    if len(new_row_diffs) != num_pairs:
                        continue
                    
                    # Check global duplicates
                    # Global set without this row's old diffs
                    other_diffs = used_global - row_diffs
                    if new_row_diffs & other_diffs:
                        continue
                    
                    # Accept
                    row[pos] = new_val
                    used_global = (used_global - row_diffs) | new_row_diffs
                    improved = True
                    break
    
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    rng = random.Random(42)
    
    # Phase 1: Greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Backtracking with decreasing bounds
    if time.time() < deadline - 20.0:
        # Try to find a tighter bound
        for max_val in range(best_scope, 104, -1):
            if time.time() > deadline - 20.0:
                break
            bt_rows = _construct_with_backtracking(n_rows, k, max_val, rng, deadline - 20.0)
            if bt_rows is not None:
                scope = _scope(bt_rows)
                if scope < best_scope:
                    best_rows = bt_rows
                    best_scope = scope
                    break # Found valid at this max, try lower? Or just accept?
                    # Actually, if we found one at max_val, it might not be the min.
                    # But backtracking is slow. Let's just take the first valid one found
                    # and then refine with local search.
    
    # Phase 3: Local Search
    if time.time() < deadline - 5.0:
        improved = _local_search_improve(best_rows, k, deadline - 5.0, rng)
        if _valid(improved):
            new_scope = _scope(improved)
            if new_scope < best_scope:
                best_rows = improved
                best_scope = new_scope

    return best_rows

# EVOLVE-BLOCK-END
