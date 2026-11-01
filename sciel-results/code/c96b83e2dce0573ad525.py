"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_rows = None
    best_scope = float('inf')
    
    # 1. Deterministic Greedy Initialization
    rows = _greedy_construct(n_rows, k)
    if rows:
        best_rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in rows)
        
    # 2. Local Search Refinement Loop
    # We try to improve the solution by moving elements to smaller values.
    # We alternate between refining the current best and trying randomized restarts if stuck.
    
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # Track stagnation to trigger restarts
    stagnation_count = 0
    max_stagnation = 50
    
    while time.time() < deadline:
        improved, new_rows, new_scope = _local_search(current_rows, k, deadline)
        
        if improved:
            current_rows = new_rows
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
            stagnation_count = 0
            # If we hit the theoretical lower bound, we can stop
            if best_scope <= 105:
                break
        else:
            stagnation_count += 1
            if stagnation_count >= max_stagnation:
                # Try a randomized restart
                if time.time() > deadline - 10:
                    break
                random.seed(int(time.time() * 1000) % 100000)
                rand_rows = _randomized_greedy(n_rows, k, deadline)
                if rand_rows:
                    # Refine the randomized start
                    improved_r, refined_rows, refined_scope = _local_search(rand_rows, k, deadline)
                    if refined_scope < best_scope:
                        best_scope = refined_scope
                        best_rows = [r[:] for r in refined_rows]
                        current_rows = refined_rows
                        current_scope = refined_scope
                        stagnation_count = 0
                    else:
                        # If randomized start didn't help, go back to current best
                        # and maybe perturb it? For now, just break and try another strategy
                        # or continue with current best if it's still the best.
                        pass
                else:
                    break
    
    return best_rows

def _greedy_construct(n_rows, k):
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

def _randomized_greedy(n_rows, k, deadline):
    used, rows = set(), []
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        row, x = [0], 0
        while len(row) < k + 1:
            if time.time() > deadline:
                return None
            # Try candidates in a biased random order
            # Start from x+1, look up to x+100
            candidates = list(range(x + 1, x + 100))
            random.shuffle(candidates)
            # Bias towards smaller values
            candidates.sort(key=lambda c: c + random.random() * 20)
            
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= new
                    x = cand
                    found = True
                    break
            if not found:
                # If not found in range, try linear scan up to x+500
                for cand in range(x + 100, x + 500):
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
            if not found:
                return None
        rows.append(row)
    return rows

def _local_search(rows, k, deadline):
    """
    Attempts to reduce the scope of the given rows by moving elements to smaller values.
    Returns (improved, new_rows, new_scope).
    """
    # Deep copy to avoid modifying original
    curr_rows = [r[:] for r in rows]
    n_rows = len(curr_rows)
    
    # Precompute global used differences
    # used_diffs[d] = True if d is used
    # We need to handle conflicts efficiently.
    # Since n_rows and k are small (7 rows, 6 elements -> 21 diffs per row, 147 total),
    # a set is sufficient.
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Initialize global set
    global_diffs = set()
    for row in curr_rows:
        global_diffs |= get_row_diffs(row)
        
    current_scope = max(max(r) for r in curr_rows)
    improved = False
    
    # Iterate until no improvement or deadline
    # We do passes over all elements
    while time.time() < deadline:
        pass_improved = False
        # Randomize order of elements to escape local minima
        # Create list of (row_idx, elem_idx)
        moves = []
        for i in range(n_rows):
            for j in range(1, k + 1): # j=0 is 0, fixed
                moves.append((i, j))
        random.shuffle(moves)
        
        for i, j in moves:
            if time.time() > deadline:
                return improved, curr_rows, current_scope
                
            row = curr_rows[i]
            if j >= len(row):
                continue
                
            current_val = row[j]
            min_val = row[j-1] + 1
            max_val = row[j+1] - 1 if j + 1 < len(row) else current_val - 1
            
            # We want to try values smaller than current_val
            # Try from min_val up to current_val - 1
            # To be efficient, we can try random values in [min_val, current_val-1] or just scan down
            # Scanning down from current_val - 1 is often good for "tightening"
            
            # Optimization: Check if we can just move to min_val?
            # Or try a few random candidates?
            # Let's try scanning from current_val - 1 down to min_val.
            # This is O(N) per move, but N is small.
            
            # Remove current differences of row i from global set
            # We need to be careful: we remove diffs involving index j, and diffs within the row that might change?
            # Actually, moving row[i][j] changes ALL diffs involving j.
            # Diffs NOT involving j remain the same.
            
            # Let's compute the set of diffs for row i that DO NOT involve j.
            # And the set of diffs that DO involve j.
            
            # Diffs not involving j:
            # Pairs (a, b) where a != j and b != j.
            # Diffs involving j:
            # Pairs (j, b) where b > j
            # Pairs (a, j) where a < j
            
            # Let's just recompute the whole row's diffs for the candidate value.
            # Since k is small (6), row length is 7. Diffs per row = 21.
            # Recomputing is cheap.
            
            # Remove all diffs of row i from global_diffs
            old_diffs = get_row_diffs(row)
            global_diffs -= old_diffs
            
            # Try candidates
            # We prioritize smaller values
            candidates = range(current_val - 1, min_val - 1, -1)
            
            best_candidate = None
            for cand in candidates:
                # Construct new row
                new_row = row[:]
                new_row[j] = cand
                
                # Check strictly increasing
                if cand <= new_row[j-1]:
                    continue
                if j + 1 < len(new_row) and cand >= new_row[j+1]:
                    continue
                
                # Compute new diffs
                new_diffs = get_row_diffs(new_row)
                
                # Check for internal duplicates (Sidon property)
                # If len(new_diffs) < 21, there's a duplicate
                if len(new_diffs) != k * (k + 1) // 2:
                    continue
                
                # Check against global_diffs
                if new_diffs & global_diffs:
                    continue
                
                # Valid!
                best_candidate = cand
                # We found the smallest valid candidate by scanning down, so we can break
                # Actually, we are scanning from high to low, so the first one found is the smallest?
                # No, range(current_val - 1, min_val - 1, -1) goes from high to low.
                # So the first valid one is the LARGEST valid one < current_val.
                # We want the SMALLEST valid one to maximize scope reduction?
                # No, we want to reduce the scope. The scope is max(all elements).
                # If we move an element to a smaller value, it helps if that element was the max.
                # Even if it's not the max, reducing it might allow other elements to be reduced later.
                # So yes, we want the smallest possible value.
                # So we should scan from min_val UP to current_val - 1?
                # But that's slower.
                # Let's just find ANY valid smaller value first to ensure progress, then refine?
                # Or scan from min_val to current_val - 1?
                # Given the small range, scanning up is fine.
                
                # Let's restart the candidate loop logic in my head.
                # I want the smallest valid value.
                # So I should iterate from min_val to current_val - 1.
                # Let's break and re-do the candidate selection properly.
                break 

            # The above loop was flawed for finding the MINIMUM. Let's do it properly.
            best_candidate = None
            for cand in range(min_val, current_val):
                new_row = row[:]
                new_row[j] = cand
                
                if cand <= new_row[j-1]:
                    continue
                if j + 1 < len(new_row) and cand >= new_row[j+1]:
                    continue
                
                new_diffs = get_row_diffs(new_row)
                
                if len(new_diffs) != k * (k + 1) // 2:
                    continue
                
                if new_diffs & global_diffs:
                    continue
                
                best_candidate = cand
                break # Found the smallest valid one
            
            if best_candidate is not None:
                # Apply move
                row[j] = best_candidate
                curr_rows[i] = row
                global_diffs |= get_row_diffs(row)
                
                # Update scope
                new_scope = max(max(r) for r in curr_rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                
                improved = True
                pass_improved = True
            else:
                # No move found, put old diffs back
                global_diffs |= old_diffs
            
        if not pass_improved:
            break
            
    return improved, curr_rows, current_scope

# EVOLVE-BLOCK-END
