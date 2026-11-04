"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return set of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _total_scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used

def _greedy_row(used, k):
    """
    Greedily construct a row of k+1 integers starting at 0.
    Each new element is the smallest integer > last that doesn't create
    a difference already in 'used'.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs

def _greedy_construct(n_rows, k):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, new_diffs = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= new_diffs
    return rows

def _randomized_greedy_row(used, k, rng, pool_size=5):
    """
    Randomized greedy: at each step, pick from the top 'pool_size' smallest
    valid candidates with some randomness.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        if rng.random() < 0.6:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, new_diffs

def _randomized_greedy(n_rows, k, rng, pool_size=5, deadline=None):
    """Randomized greedy construction with multiple restarts."""
    best_rows = None
    best_scope = float('inf')
    
    max_restarts = 100
    if deadline is None:
        deadline = time.time() + 100
        
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row, new_diffs = _randomized_greedy_row(used, k, rng, pool_size)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= new_diffs
        if success:
            scope = _total_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    return best_rows if best_rows is not None else _greedy_construct(n_rows, k)

def _is_valid(rows):
    """Check if all positive differences across all rows are distinct."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _local_search_shrink(rows, k, rng, deadline):
    """
    Structural change: Targeted Local Search.
    Focuses on reducing the scope by shrinking elements, especially the largest ones.
    Uses a 'used' set for O(1) conflict checks.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # We will iterate until deadline or no improvements
    # To avoid infinite loops on stagnation, we use a stagnation counter
    stagnation_count = 0
    max_stagnation = 5000
    
    while time.time() < deadline and stagnation_count < max_stagnation:
        improved = False
        
        # 1. Try to shrink the largest element of the row with the largest scope
        # Find the row with the max last element
        max_row_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        row = rows[max_row_idx]
        
        # Try to shrink the last element
        last_idx = len(row) - 1
        old_val = row[last_idx]
        
        # Search for a smaller valid value for row[last_idx]
        # It must be > row[last_idx-1]
        lower_bound = row[last_idx-1] + 1
        found = False
        for new_val in range(old_val - 1, lower_bound - 1, -1):
            # Calculate diffs involving old_val and new_val
            # Diffs are: val - row[i] for i < last_idx
            old_diffs = {old_val - row[i] for i in range(last_idx)}
            new_diffs = {new_val - row[i] for i in range(last_idx)}
            
            # Check conflicts with global used set excluding old_diffs
            # We can do this by checking if new_diffs contains anything in (used - old_diffs)
            # Optimization: if new_diffs is subset of old_diffs, no conflict with global (other than itself which is removed)
            # But we must ensure new_diffs don't conflict with *other* rows' diffs.
            
            # Efficient check:
            # 1. Remove old_diffs from a temporary view? No, just check intersection.
            # 2. If any d in new_diffs is in used AND d is NOT in old_diffs -> Conflict.
            
            # Note: new_diffs might contain diffs that were in old_diffs? 
            # If new_val < old_val, new_diffs are generally smaller.
            # It is possible that new_diffs intersects old_diffs (e.g. if row is [0, 10, 20], old_val=20, new_val=15. 
            # old_diffs={20,10}, new_diffs={15,5}. Intersection empty.
            # If row is [0, 5, 10], old_val=10, new_val=6. old_diffs={10,5}, new_diffs={6,1}. Intersection empty.
            # Actually, since row is increasing, if new_val < old_val, new_diffs < old_diffs component-wise?
            # row[i] is fixed. new_val - row[i] < old_val - row[i].
            # So new_diffs and old_diffs are disjoint sets of integers? Not necessarily disjoint values, but distinct positions.
            # However, a value d can be in both if e.g. old_val - row[0] == new_val - row[1].
            # So we must check carefully.
            
            conflict = False
            for d in new_diffs:
                if d in used:
                    # If d was in old_diffs, it's "freed" by the move.
                    # But wait, 'used' contains old_diffs.
                    # So if d is in used, it might be from this row (old) or another row.
                    # If it's from this row (old), it's valid to reuse it? No, because we are replacing the element.
                    # The diff d is removed from the global set. If the new element creates the same diff d, it's fine, 
                    # UNLESS another row also uses d.
                    # So: Valid if (new_diffs - old_diffs) is disjoint from (used - old_diffs).
                    # Which is equivalent to: new_diffs is disjoint from (used - old_diffs).
                    if d not in old_diffs:
                        conflict = True
                        break
            
            if not conflict:
                # Valid move
                row[last_idx] = new_val
                used = (used - old_diffs) | new_diffs
                new_scope = _total_scope(rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                improved = True
                found = True
                break
        
        if not found:
            # Try to shrink other elements in the worst row, or other rows
            # To save time, we only do this if we haven't improved in a while
            if stagnation_count % 10 == 0:
                # Try to shrink a random element in a random row
                # This is expensive, so we limit it
                pass # Skipping complex inner loop for now to prioritize speed of the main shrink

        if not improved:
            stagnation_count += 1
            # If stuck, try a random perturbation to escape local optimum
            if stagnation_count % 100 == 0:
                # Pick a random row and a random element (not first)
                r_idx = rng.randint(0, len(rows) - 1)
                e_idx = rng.randint(1, len(rows[r_idx]) - 1)
                old_val = rows[r_idx][e_idx]
                
                # Try to increase it slightly to change the difference set
                # We want to free up small differences or break a block
                # Let's try a small increase
                for delta in [1, 2, 3, 5]:
                    new_val = old_val + delta
                    # Check bounds
                    if new_val <= rows[r_idx][e_idx-1]:
                        continue
                    if e_idx < len(rows[r_idx]) - 1 and new_val >= rows[r_idx][e_idx+1]:
                        continue
                    
                    # Calculate diffs
                    old_diffs = set()
                    new_diffs = set()
                    for i in range(len(rows[r_idx])):
                        if i == e_idx:
                            continue
                        if i < e_idx:
                            old_diffs.add(old_val - rows[r_idx][i])
                            new_diffs.add(new_val - rows[r_idx][i])
                        else:
                            old_diffs.add(rows[r_idx][i] - old_val)
                            new_diffs.add(rows[r_idx][i] - new_val)
                    
                    # Check conflict: (new_diffs - old_diffs) vs (used - old_diffs)
                    conflict = False
                    for d in new_diffs:
                        if d in used and d not in old_diffs:
                            conflict = True
                            break
                    
                    if not conflict:
                        rows[r_idx][e_idx] = new_val
                        used = (used - old_diffs) | new_diffs
                        # This move increases scope, so it doesn't improve best_scope directly,
                        # but it changes the landscape.
                        # Reset stagnation counter to allow more shrinking attempts
                        stagnation_count = 0
                        improved = True # Mark as "action taken" to avoid infinite tight loop
                        break

    return best_rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    
    # Start with greedy
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        # Fallback: very simple construction
        rows = []
        used = set()
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
    
    best_rows = rows
    best_scope = _total_scope(rows)
    
    # Deadline management
    # We have 120s total. We want to leave 5s for safety.
    main_deadline = time.time() + 115
    
    # Phase 1: Randomized Greedy Restarts
    # We allocate 20% of the time to finding good initial solutions
    rg_deadline = time.time() + (115 * 0.2)
    
    # Run randomized greedy in parallel? No, single thread.
    # We can run it for a fixed number of restarts or until time.
    # Let's run it until rg_deadline
    while time.time() < rg_deadline:
        rand_rows = _randomized_greedy(n_rows, k, rng, pool_size=rng.choice([3, 5, 8]), deadline=rg_deadline)
        if rand_rows is not None:
            scope = _total_scope(rand_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rand_rows]
        # The _randomized_greedy function itself has a loop, 
        # but we call it in a loop here to allow different rng states?
        # Actually, _randomized_greedy runs multiple restarts internally.
        # To be safe, let's just call it once with a deadline, 
        # and if it finishes early, we can restart the process?
        # The internal loop of _randomized_greedy breaks on deadline.
        # So this outer while might only run once or twice.
        # Let's modify: just run _randomized_greedy with the deadline.
        # It will do as many restarts as it can.
        break 

    # Phase 2: Local Search on the best solution found
    # Use remaining time
    ls_deadline = time.time() + (115 * 0.8)
    # Ensure we don't exceed main deadline
    ls_deadline = min(ls_deadline, main_deadline)
    
    if time.time() < ls_deadline:
        # Run local search
        ls_rows = _local_search_shrink(best_rows, k, rng, ls_deadline)
        ls_scope = _total_scope(ls_rows)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = [r[:] for r in ls_rows]
            
    return best_rows

# EVOLVE-BLOCK-END
