import time
import sys
import random

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # 1. Get an initial valid solution using greedy
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        # Fallback: return empty or try hard? Task says valid when... 
        # If greedy fails, we might return a partial or invalid? 
        # The contract says "Valid when...". If we return invalid, score is 0.
        # Greedy should always work for small n,k.
        return []
    
    best_scope = max(max(r) for r in best_solution)
    
    # 2. Try to improve by searching for solutions with smaller max scope
    # We know the theoretical lower bound is roughly total_diffs
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Start checking from current best_scope - 1 downwards
    # Or start from a reasonable lower bound and go up?
    # Going down from best_scope is safer because we already have a valid solution.
    # If we find a solution with scope S, we update best_scope = S.
    
    # To make the search effective, we should randomize the order of rows or choices.
    # We will run multiple attempts.
    
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Try to find a solution with max < best_scope
        target_scope = best_scope - 1
        
        # We can also try to jump down, e.g., to total_diffs
        # But step-by-step is safer to not miss small improvements.
        # However, if the gap is large, step-by-step is slow.
        # Let's try to find *any* solution with max <= target_scope.
        
        # Use a randomized backtracking solver
        # The solver will try to construct rows such that all elements <= target_scope
        # and all diffs are unique.
        
        solution = _backtrack_solve_optimized(n_rows, k, target_scope, deadline)
        
        if solution:
            curr_scope = max(max(r) for r in solution)
            if curr_scope < best_scope:
                best_scope = curr_scope
                best_solution = solution
            # If we found a solution with exactly target_scope, 
            # we can try to go lower in the next iteration.
            # If we found a solution with scope < target_scope, 
            # best_scope is already updated.
        else:
            # If we couldn't find a solution with max <= target_scope,
            # it might be due to search failure (incomplete search) or impossibility.
            # To avoid getting stuck in a local search failure, 
            # we can try a different random seed or strategy.
            # For now, we just break or continue with randomization.
            # If we fail to find a solution for target_scope, 
            # it's likely impossible or hard. 
            # Let's try to improve the current best_solution via local search instead.
            improved = _local_search_improve(best_solution, n_rows, k, deadline)
            if improved:
                curr_scope = max(max(r) for r in improved)
                if curr_scope < best_scope:
                    best_scope = curr_scope
                    best_solution = improved
                else:
                    # No improvement, break to avoid wasting time
                    break
    
    return best_solution

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Search for next element
            # We can limit the search range to avoid infinite loops if impossible
            # But for valid instances, it should find one.
            for x in range(current_max + 1, current_max + 200):
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _backtrack_solve_optimized(n_rows, k, max_scope, deadline):
    """
    Try to construct n_rows rows of size k+1, starting with 0,
    such that all elements <= max_scope and all diffs are distinct.
    """
    used_diffs = set()
    rows = []
    
    # Pre-calculate required diffs
    diffs_per_row = k * (k + 1) // 2
    total_required = n_rows * diffs_per_row
    
    def backtrack(row_idx, row, current_max):
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            return True
        
        # If this is the first element of a new row, it's 0
        if len(row) == 0:
            row.append(0)
            # Recurse to fill the rest of the row
            return backtrack(row_idx, row, 0)
        
        if len(row) == k + 1:
            # Row is complete
            rows.append(row)
            # Move to next row
            return backtrack(row_idx + 1, [], 0)
        
        # Determine next position in row
        pos = len(row)
        
        # Pruning: Check if enough diffs remain
        # Diffs used so far:
        # Current row has pos elements, so pos*(pos-1)/2 diffs? No.
        # Diffs are between any pair.
        # Let's just count used_diffs size.
        used_count = len(used_diffs)
        remaining_rows = n_rows - row_idx
        # Diffs needed for current row completion:
        # Current row has `pos` elements. Total elements k+1.
        # Diffs already in current row: pos*(pos-1)/2
        # Total diffs for a full row: k*(k+1)/2
        # Diffs needed to complete current row: total_row_diffs - pos*(pos-1)/2
        # Diffs needed for remaining rows: remaining_rows * total_row_diffs
        # Total remaining diffs needed:
        # Note: used_diffs includes diffs from previous rows AND current row.
        # So remaining capacity = max_scope - used_count (approx, since diffs are <= max_scope)
        # Actually, diffs are in 1..max_scope.
        # Available diffs = max_scope - len(used_diffs)
        # Required diffs = (total_required - used_count)
        # If available < required, prune.
        
        required_remaining = total_required - used_count
        available = max_scope - used_count
        
        if available < required_remaining:
            return False
            
        # Heuristic: Try smaller values first to keep scope low
        # But we must ensure we don't exceed max_scope
        start_val = row[-1] + 1
        end_val = min(max_scope, row[-1] + 50) # Limit search window for speed
        
        # To introduce randomness and escape local minima, 
        # we can shuffle the candidate range or pick randomly.
        # For now, let's try sequential, but with a random start offset?
        # No, sequential is good for finding *a* solution.
        # Randomization helps if sequential fails due to ordering.
        
        candidates = list(range(start_val, end_val + 1))
        # Add some larger candidates if we are near the end?
        # If we are close to max_scope, we might need larger jumps? No, we want small.
        
        # Randomize candidates to avoid deterministic failure patterns
        random.shuffle(candidates)
        
        for x in candidates:
            if x > max_scope:
                continue
                
            # Check validity
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Tentatively add
            row.append(x)
            used_diffs.update(new_diffs)
            
            if backtrack(row_idx, row, x):
                return True
            
            # Backtrack
            row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False
    
    return backtrack(0, [], 0)

def _local_search_improve(rows, n_rows, k, deadline):
    """
    Try to reduce the max element of the solution.
    """
    current_rows = [r[:] for r in rows]
    used_diffs = set()
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used_diffs.add(r[j] - r[i])
                
    best_scope = max(max(r) for r in current_rows)
    
    # Try to reduce the global max
    # Find the row with the max element
    # Try to replace that element with a smaller one
    
    for _ in range(10): # Limit iterations
        if time.time() > deadline - 5:
            break
            
        # Find the global max element and its location
        max_val = -1
        max_row_idx = -1
        max_pos = -1
        for i, r in enumerate(current_rows):
            for j, val in enumerate(r):
                if val > max_val:
                    max_val = val
                    max_row_idx = i
                    max_pos = j
                    
        # Try to reduce max_val
        row = current_rows[max_row_idx]
        pos = max_pos
        old_val = row[pos]
        
        if old_val == 0:
            break # Can't reduce 0
            
        # Collect diffs involving old_val
        diffs_to_remove = set()
        for i in range(pos):
            diffs_to_remove.add(old_val - row[i])
        for j in range(pos+1, len(row)):
            diffs_to_remove.add(row[j] - old_val)
            
        forbidden_diffs = used_diffs - diffs_to_remove
        
        # Try values from row[pos-1]+1 to old_val-1
        # We want the largest possible value < old_val that is valid?
        # No, we want to reduce the max. Any valid value < old_val works.
        # To be safe, maybe try the largest valid one < old_val to keep the row "tight"?
        # Or smallest?
        # Let's try from old_val - 1 downwards.
        
        found = False
        for new_val in range(old_val - 1, row[pos-1], -1):
            valid = True
            new_diffs = set()
            for i in range(pos):
                d = new_val - row[i]
                if d in forbidden_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid: continue
            
            for j in range(pos+1, len(row)):
                d = row[j] - new_val
                if d in forbidden_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
                
            if valid:
                # Apply change
                used_diffs.difference_update(diffs_to_remove)
                used_diffs.update(new_diffs)
                row[pos] = new_val
                found = True
                break
                
        if not found:
            # Try to move a different element?
            # For simplicity, break if we can't reduce the global max
            break
            
    return current_rows
