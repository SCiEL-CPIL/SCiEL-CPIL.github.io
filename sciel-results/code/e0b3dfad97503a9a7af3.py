import time
import random
import sys

def _greedy_fallback(n_rows, k):
    """Greedy construction as fallback."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _scope(rows):
    return max(max(row) for row in rows)

def _validate(rows):
    """Check if all differences across all rows are distinct."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _compute_row_diffs(row):
    """Compute the set of differences within a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _construct_with_bound(n_rows, k, max_scope, rng, deadline):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all differences are distinct and max entry <= max_scope.
    Returns None if not found within deadline.
    """
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, row, pos, max_scope, used_diffs, deadline):
        if time.time() > deadline:
            return False
        
        if pos == k + 1:
            rows.append(row[:])
            return True
        
        if row_idx == n_rows:
            return True
        
        # Determine range of candidates for row[pos]
        lower = row[-1] + 1 if row else 1
        upper = min(max_scope, lower + 100)  # Cap search for efficiency
        
        # Generate candidates
        candidates = list(range(lower, upper + 1))
        
        # Shuffle with bias towards smaller values for diversity
        if rng.random() < 0.3:
            rng.shuffle(candidates)
        else:
            # Partial shuffle: keep small values more likely
            for i in range(len(candidates) - 1, 0, -1):
                if rng.random() < 0.5:
                    candidates[i], candidates[i - 1] = candidates[i - 1], candidates[i]
        
        for cand in candidates:
            # Check if cand is valid
            new_diffs = set()
            valid = True
            
            for i in range(len(row)):
                d = cand - row[i]
                if d <= 0 or d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Also check internal consistency (should be fine since row is increasing)
            # But we need to ensure new_diffs don't conflict with each other (they won't since row is increasing)
            
            # Try this candidate
            row.append(cand)
            old_used = used_diffs.copy()
            used_diffs |= new_diffs
            
            if backtrack(row_idx, row, pos + 1, max_scope, used_diffs, deadline):
                return True
            
            # Backtrack
            row.pop()
            used_diffs = old_used
        
        return False
    
    # Try to build all rows
    for i in range(n_rows):
        row = [0]
        if not backtrack(i, row, 1, max_scope, used_diffs, deadline):
            return None
    
    return rows

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Local search to reduce scope.
    Strategy: Iteratively try to decrease each entry to the minimum valid value.
    """
    best = [row[:] for row in rows]
    best_scope = _scope(best)
    
    def get_other_diffs(exclude_idx):
        other_diffs = set()
        for i, row in enumerate(best):
            if i == exclude_idx:
                continue
            other_diffs |= _compute_row_diffs(row)
        return other_diffs
    
    improved = True
    
    while time.time() < deadline:
        improved = False
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for r_idx in row_order:
            row = best[r_idx]
            other_diffs = get_other_diffs(r_idx)
            
            for p in range(len(row) - 1, 0, -1):
                current_val = row[p]
                lower_bound = row[p - 1] + 1
                
                if current_val == lower_bound:
                    continue
                
                # Compute static diffs (not involving position p)
                static_diffs = set()
                for i in range(len(row)):
                    if i == p:
                        continue
                    for j in range(i + 1, len(row)):
                        if j == p:
                            continue
                        static_diffs.add(row[j] - row[i])
                
                best_candidate = current_val
                found_better = False
                
                for cand in range(lower_bound, current_val):
                    new_diffs = set()
                    valid = True
                    
                    for i in range(p):
                        d = cand - row[i]
                        if d in other_diffs or d in new_diffs or d in static_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    for j in range(p + 1, len(row)):
                        d = row[j] - cand
                        if d in other_diffs or d in new_diffs or d in static_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        row[p] = cand
                        best_candidate = cand
                        found_better = True
                        break
                
                if found_better:
                    improved = True
        
        if not improved:
            break
    
    return best, _scope(best)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # 1. Get initial greedy solution
    greedy_sol = _greedy_fallback(n_rows, k)
    best_rows = greedy_sol
    best_scope = _scope(greedy_sol)
    
    # 2. Try iterative deepening with bound
    # Start from a reasonable bound and decrease
    # The counting bound is n_rows * k*(k+1)/2 = 7*15 = 105
    # Greedy gives something higher
    
    # Try to find solutions with decreasing bounds
    current_bound = best_scope
    min_bound = 105  # Theoretical lower bound
    
    rng = random.Random(123)
    
    # Try a few different bounds
    for bound in range(current_bound, min_bound - 1, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple times with different random seeds
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            sol = _construct_with_bound(n_rows, k, bound, rng, deadline - 30)
            
            if sol is not None:
                sol_scope = _scope(sol)
                if sol_scope < best_scope:
                    best_scope = sol_scope
                    best_rows = sol
                
                # If we found a solution at the theoretical minimum, stop
                if best_scope == min_bound:
                    return best_rows
        
        # If we found a solution at this bound, try a smaller bound
        # Otherwise, increase the bound (shouldn't happen since we're decreasing)
    
    # 3. Local search on the best solution found
    if time.time() < deadline - 10:
        local_sol, local_scope = _local_search(best_rows, n_rows, k, deadline - 5, rng)
        if local_scope < best_scope:
            best_scope = local_scope
            best_rows = local_sol
    
    return best_rows
