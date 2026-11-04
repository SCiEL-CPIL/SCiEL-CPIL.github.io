"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _row_diffs(row):
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
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, randomize=False):
    """Greedy construction with optional randomization."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                # Try a random step within reasonable bounds
                if row:
                    max_step = max(1, int((k + 1) * 1.5))
                    for _attempt in range(20):
                        step = random.randint(1, max_step)
                        x = row[-1] + step
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                    else:
                        # Fallback to sequential
                        x = row[-1] + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used:
                                break
                            x += 1
                else:
                    x = 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _backtrack_row(used, k, max_val, deadline, random_order=False):
    """Backtrack search for a single row of length k+1 starting at 0,
    max entry <= max_val, with all differences disjoint from used."""
    row = [0]
    row_diffs = set()
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        
        # Generate candidate values
        candidates = list(range(row[-1] + 1, max_val + 1))
        if random_order:
            random.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            if new_diffs & row_diffs:
                continue
            
            row.append(x)
            row_diffs.update(new_diffs)
            if backtrack(pos + 1):
                return True
            row_diffs.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if backtrack(1):
        return list(row)
    return None


def _construct_with_backtracking(n_rows, k, deadline):
    """Construct rows one by one using backtracking, minimizing max entry."""
    used = set()
    rows = []
    
    for r_idx in range(n_rows):
        # Determine what max value we need to search up to
        # Start with a reasonable estimate
        current_max = max((max(r) for r in rows), default=0)
        search_max = current_max + k + 10  # Give some room
        
        # Try to find a row with minimal max entry
        found = None
        # Try smaller max values first
        for target_max in range(current_max + 1, search_max + 1):
            if time.time() > deadline - 5.0:
                break
            row = _backtrack_row(used, k, target_max, deadline - 5.0, random_order=(r_idx % 2 == 1))
            if row is not None:
                found = row
                break
        
        if found is None:
            # Fallback to greedy
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            found = row
        
        rows.append(found)
        used |= _row_diffs(found)
    
    return rows


def _local_search_improve(rows, k, deadline):
    """Local search: try to reduce max entry by modifying individual elements with incremental diff checks."""
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    
    # Precompute global used diffs
    used = set()
    for r in best:
        used |= _row_diffs(r)
    
    # Helper to get diffs for a specific row
    def get_row_diffs(row):
        return _row_diffs(row)

    while time.time() < deadline - 2.0:
        improved = False
        current_scope = _scope(best)
        
        # Find the row with the maximum entry
        max_row_idx = max(range(len(best)), key=lambda i: max(best[i]))
        
        # Try to reduce elements in that row from the end
        row = best[max_row_idx]
        for pos in range(len(row) - 1, 0, -1):
            if time.time() > deadline - 2.0:
                break
            old_val = row[pos]
            lower_bound = row[pos - 1] + 1
            
            # Try values from lower_bound to old_val - 1
            # We can optimize by checking if new_val < old_val is even possible to reduce scope
            # If pos is not the last element, reducing it doesn't reduce scope unless it IS the max
            # But reducing any element might allow other reductions later.
            # However, to minimize scope, we focus on the max element first.
            
            for new_val in range(lower_bound, old_val):
                if time.time() > deadline - 2.0:
                    break
                
                # Calculate old diffs involving pos
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(old_val - row[i])
                
                # Calculate new diffs involving pos
                new_diffs = set()
                for i in range(pos):
                    new_diffs.add(new_val - row[i])
                
                # The set of diffs that remain in the global set if we make this change:
                # (used - old_diffs) | new_diffs
                # We need to ensure (used - old_diffs) and new_diffs are disjoint.
                # i.e., new_diffs & (used - old_diffs) == empty
                
                # Optimization: 
                # 1. new_diffs must not contain any diff that is in used but NOT in old_diffs
                # 2. new_diffs must not contain any diff that is in old_diffs? 
                #    Wait, if a diff is in both old_diffs and new_diffs, it's fine.
                #    So we just need: (new_diffs & used) - old_diffs == empty
                #    Or: new_diffs & (used - old_diffs) == empty
                
                conflict = new_diffs & (used - old_diffs)
                if conflict:
                    continue
                
                # Also need to check if new_diffs has internal conflicts? 
                # No, new_diffs are just differences from new_val to previous elements.
                # They are distinct if row is strictly increasing, which it is.
                
                # Perform the change
                row[pos] = new_val
                
                # Update used set
                used -= old_diffs
                used |= new_diffs
                
                # Check if scope improved
                new_scope = _scope(best)
                if new_scope < best_scope:
                    best_scope = new_scope
                    improved = True
                    break
                else:
                    # Revert
                    row[pos] = old_val
                    used -= new_diffs
                    used |= old_diffs
            
            if improved:
                break
        
        if not improved:
            # If no simple reduction found, try a perturbation
            # Randomly pick a row and a position, and try a random valid replacement
            # This helps escape local minima
            if random.random() < 0.7:
                ridx = random.randint(0, len(best) - 1)
                r = best[ridx]
                # Pick a position to change, preferably not the first
                pos = random.randint(1, len(r) - 1)
                old_val = r[pos]
                
                # Try a random new value that keeps row increasing
                # Range: r[pos-1]+1 to r[pos+1]-1 (if pos+1 exists) or some upper bound
                low = r[pos-1] + 1
                high = r[pos+1] - 1 if pos + 1 < len(r) else old_val + 10
                
                if low < high:
                    # Try a few random values
                    for _ in range(5):
                        new_val = random.randint(low, high)
                        if new_val == old_val:
                            continue
                        
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - r[i])
                        for i in range(pos + 1, len(r)):
                            old_diffs.add(r[i] - old_val)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - r[i])
                        for i in range(pos + 1, len(r)):
                            new_diffs.add(r[i] - new_val)
                        
                        conflict = new_diffs & (used - old_diffs)
                        if not conflict:
                            # Accept this perturbation (even if scope doesn't improve immediately)
                            # This is a "sideways" move to explore
                            r[pos] = new_val
                            used -= old_diffs
                            used |= new_diffs
                            break
            else:
                # If no improvement and no perturbation accepted, break to avoid infinite loop
                # Or try another restart logic in main construct
                break
    
    return best, best_scope


def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s kill
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Backtracking construction (try to do better)
    if time.time() < deadline - 30.0:
        bt_rows = _construct_with_backtracking(n_rows, k, deadline - 30.0)
        bt_scope = _scope(bt_rows)
        if bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Local search improvement (Optimized)
    if time.time() < deadline - 15.0:
        ls_rows, ls_scope = _local_search_improve(best_rows, k, deadline - 15.0)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
    
    # Phase 4: Try multiple random restarts if time permits
    # Since local search is faster now, we can afford more restarts
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 100:
        attempts += 1
        # Random greedy
        rrows = _greedy_construct(n_rows, k, randomize=True)
        rs = _scope(rrows)
        
        # Only run local search if the greedy start is promising
        # or if we have plenty of time
        if rs < best_scope or (time.time() < deadline - 20.0 and rs < best_scope + 5):
            # Quick local search on the new candidate
            if time.time() < deadline - 5.0:
                ls2, ls2s = _local_search_improve(rrows, k, deadline - 5.0)
                if ls2s < best_scope:
                    best_rows = ls2
                    best_scope = ls2s
    
    # Phase 5: Final local search pass on the best found
    if time.time() < deadline - 2.0:
        ls_rows, ls_scope = _local_search_improve(best_rows, k, deadline - 1.0)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
    
    return best_rows

# EVOLVE-BLOCK-END
