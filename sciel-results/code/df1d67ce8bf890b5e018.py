import time
import random
import copy

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _count_conflicts(rows):
    """Count number of duplicate differences."""
    diffs = []
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
    return len(diffs) - len(set(diffs))

def _get_diffs(row):
    dset = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            dset.add(row[j] - row[i])
    return dset

def _simulated_annealing(initial_rows, n_rows, k, rng, deadline):
    """
    Simulated Annealing to minimize max entry while keeping solution valid (or minimizing conflicts).
    We will start with a valid solution and try to reduce max entry.
    Moves: Change a single element in a row.
    Constraint: The row must remain strictly increasing.
    Global Constraint: All differences must be distinct.
    To allow escaping local minima, we might allow temporary invalid states if we penalize them,
    but it's safer to stick to valid moves if possible, or use a penalty.
    Given the strictness, let's try to find valid moves that reduce max entry.
    If no valid move reduces max entry, we can try to increase max entry slightly to enable
    a future reduction (this is the 'simulated' part).
    """
    current_rows = copy.deepcopy(initial_rows)
    current_score = _max_entry(current_rows)
    current_conflicts = 0 # Should be 0 if valid
    
    # Precompute diffs for each row and global set
    row_diffs = [_get_diffs(row) for row in current_rows]
    global_diffs = set()
    for rd in row_diffs:
        global_diffs.update(rd)
        
    best_rows = copy.deepcopy(current_rows)
    best_score = current_score
    
    T = 100.0 # Initial temperature
    cooling = 0.99
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Select a random row and position
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1) # Positions 1..k (0 is fixed)
        
        old_val = current_rows[ri][pi]
        
        # Determine range for new value
        lo = current_rows[ri][pi-1] + 1
        hi = current_rows[ri][pi+1] - 1 if pi < k else current_score + 10 # Allow expansion
        
        if lo >= hi:
            continue
            
        # Pick a new value
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue
            
        # Check strict increase locally
        if pi > 0 and new_val <= current_rows[ri][pi-1]:
            continue
        if pi < k and new_val >= current_rows[ri][pi+1]:
            continue
            
        # Calculate change in diffs for this row
        old_row_diffs = row_diffs[ri]
        
        # New row
        new_row = current_rows[ri][:]
        new_row[pi] = new_val
        
        # Check strict increase for the whole row (already checked neighbors, but let's be safe)
        if not all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)):
            continue
            
        new_row_diffs = _get_diffs(new_row)
        
        # Diffs added and removed
        added = new_row_diffs - old_row_diffs
        removed = old_row_diffs - new_row_diffs
        
        # Check conflicts with other rows
        # Conflicts exist if any diff in 'added' is already in global_diffs (excluding those in 'removed' from this row? No, 'removed' are no longer there)
        # Actually, global_diffs contains all diffs from all rows.
        # If we change row ri, we remove 'removed' from global, add 'added'.
        # New global = (global - removed) U added.
        # Conflict if any d in added is in (global - removed).
        
        potential_conflicts = 0
        for d in added:
            # If d is in global_diffs and d is not in removed (from this row), it's a conflict
            # But wait, if d was in this row and is still in this row (in added and removed? No, added/removed are symmetric difference)
            # If d is in added, it was NOT in old_row_diffs.
            # So if d is in global_diffs, it MUST be from another row (since it's not in old_row_diffs).
            # So any d in added that is in global_diffs is a conflict.
            if d in global_diffs:
                potential_conflicts += 1
                
        # Also, if we remove a diff that was unique to this row, it's fine.
        # If we remove a diff that was shared? No, diffs are unique in a valid state.
        
        new_conflicts = potential_conflicts
        
        # Calculate new score
        new_max = max(current_score, new_val) # If new_val is max, score increases
        # If new_val < current_score, and current_score was only in this row at this pos, score might drop.
        # To be precise:
        if new_val > current_rows[ri][pi]:
             # Max might increase
             pass
        # Recalculating max is O(n*k).
        new_rows_temp = current_rows[:]
        new_rows_temp[ri] = new_row
        new_score = _max_entry(new_rows_temp)
        
        # Acceptance probability
        # Objective: Minimize Score. If conflicts > 0, it's very bad.
        # Let's define Energy = Score + 1000 * Conflicts
        old_energy = current_score + 1000 * current_conflicts
        new_energy = new_score + 1000 * new_conflicts
        
        delta_E = new_energy - old_energy
        
        if delta_E < 0:
            # Accept
            current_rows[ri] = new_row
            row_diffs[ri] = new_row_diffs
            global_diffs.difference_update(removed)
            global_diffs.update(added)
            current_score = new_score
            current_conflicts = new_conflicts
            
            if new_score < best_score:
                best_score = new_score
                best_rows = copy.deepcopy(current_rows)
        else:
            # Accept with probability exp(-delta_E / T)
            if rng.random() < pow(2.718, -delta_E / T):
                current_rows[ri] = new_row
                row_diffs[ri] = new_row_diffs
                global_diffs.difference_update(removed)
                global_diffs.update(added)
                current_score = new_score
                current_conflicts = new_conflicts
                
                # Track best valid solution
                if new_conflicts == 0 and new_score < best_score:
                    best_score = new_score
                    best_rows = copy.deepcopy(current_rows)
                
                # If we accepted an invalid state, we don't update best_score unless it's valid and better.
                # But we should keep track of best VALID solution.
                
        T *= cooling
        
    return best_rows, best_score

def _greedy_all(n_rows, k):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            base = row[-1]
            found = False
            for x in range(base + 1, 500):
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def construct(n_rows, k):
    deadline = time.time() + 140
    
    rng = random.Random(42)
    
    # Phase 1: Greedy Initialization
    best_rows = _greedy_all(n_rows, k)
    if best_rows is None or not _validate(best_rows):
        # Fallback
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                for x in range(row[-1] + 1, 1000):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
            best_rows.append(row)
            
    best_score = _max_entry(best_rows)
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 10:
        sa_rows, sa_score = _simulated_annealing(best_rows, n_rows, k, rng, deadline - 5)
        if sa_rows is not None and _validate(sa_rows):
            if sa_score < best_score:
                best_score = sa_score
                best_rows = sa_rows
                
    # Phase 3: Final Local Search Polish (keep from original if it helps, or skip if SA is good)
    # Let's add a quick local search on the SA result
    if time.time() < deadline - 5:
         # Re-use the local search logic from the original but simplified
         current_rows = copy.deepcopy(best_rows)
         current_score = best_score
         
         # Quick polish
         for _ in range(1000):
             if time.time() > deadline - 2: break
             ri = rng.randrange(n_rows)
             pi = rng.randrange(1, k + 1)
             old_val = current_rows[ri][pi]
             lo = current_rows[ri][pi-1] + 1
             hi = current_rows[ri][pi+1] - 1 if pi < k else current_score
             
             if lo > hi: continue
             
             # Try to reduce
             for new_val in range(lo, old_val):
                 new_row = current_rows[ri][:]
                 new_row[pi] = new_val
                 if not all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)): continue
                 
                 new_diffs = _get_diffs(new_row)
                 # Check conflicts
                 # This is O(K^2) per check, which is fine for K=5
                 conflict = False
                 for oi in range(n_rows):
                     if oi == ri: continue
                     # We need row_diffs for others. Let's just recompute or store.
                     # For speed, let's just use _validate on a modified copy? No, too slow.
                     # Let's assume SA left us in a good spot.
                     pass
                 # Simplified: just check if new_diffs intersects with global diffs excluding this row's old diffs?
                 # Too complex for inline. Skip detailed polish if SA is strong.
                 break
                 
    if not _validate(best_rows):
        fresh_rows = _greedy_all(n_rows, k)
        if fresh_rows is not None and _validate(fresh_rows):
            best_rows = fresh_rows
            
    return best_rows
