import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer
    
    # Theoretical lower bound
    lb = n_rows * k * (k + 1) // 2
    
    # Known upper bound for (7,5) is 112.
    # We will try to find the minimum scope in [lb, 112].
    # If we don't find one, we return the best found or a valid solution with higher scope.
    
    best_result = None
    best_scope = float('inf')
    
    # We will iterate through possible scopes
    # To be efficient, we can binary search, but since the range is small (105-112), linear is fine.
    # However, if the optimal is 112, we will waste time on 105-111.
    # We can use a heuristic: try 112 first to get a valid solution, then try to improve?
    # Or just iterate from lb to 112.
    
    # Let's try to find a solution for a specific scope S.
    def find_solution(scope):
        nonlocal start_time, time_limit, best_result, best_scope
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, current_row, last_max):
            nonlocal used_diffs, rows, start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return False, None
            
            if row_idx == n_rows:
                # All rows constructed
                # The scope is the max of the last elements, which is <= scope by construction
                # But we enforced last_max ordering, so the last row's max is the global max?
                # Not necessarily, but since we ordered by last_max, the last row has the largest max.
                # So scope_actual = rows[-1][-1]
                return True, [r[:] for r in rows]
            
            if len(current_row) == k + 1:
                # Row complete
                current_max = current_row[-1]
                
                # Symmetry breaking: enforce non-decreasing max values
                if current_max < last_max:
                    return False, None
                
                rows.append(current_row)
                success, res = backtrack(row_idx + 1, [0], current_max)
                rows.pop()
                
                if success:
                    return True, res
                return False, None
            
            last_val = current_row[-1]
            
            # Determine candidates for next element x
            # x > last_val
            # x <= scope
            # Also, if this is the last element (index k), x must be >= last_max
            # If not last element, x can be anything, as long as we can reach last_max later.
            # Since we can always pick x up to scope, and last_max <= scope,
            # we only need to enforce x >= last_max if it's the last element?
            # Actually, if x < last_max, we can still pick a larger y > x later.
            # So the constraint is only on the final element of the row.
            
            is_last_elem = (len(current_row) == k)
            
            candidates = []
            for x in range(last_val + 1, scope + 1):
                if is_last_elem and x < last_max:
                    continue
                
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                candidates.append(x)
                
                # Limit branching? 
                # If scope is small, candidates are few.
                # If scope is large, candidates are many.
                # We can sort candidates to guide the search.
                # Sorting by x ascending tends to find smaller scopes faster within the bound.
                # But we are fixing the scope. So we just want *any* solution.
                # Maybe randomize or use a heuristic?
                # Let's try x ascending first.
            
            if not candidates:
                return False, None
            
            # Heuristic: Try candidates that use "large" differences?
            # Or just iterate.
            # To avoid getting stuck in a deep dead-end, we can add a small randomization
            # or try a different order.
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                
                # Update state
                old_used = used_diffs
                used_diffs = used_diffs | new_diffs
                
                success, res = backtrack(row_idx, current_row + [x], last_max)
                
                # Restore state
                used_diffs = old_used
                
                if success:
                    return True, res
            
            return False, None
        
        return backtrack(0, [0], 0)

    # Strategy:
    # 1. Try scopes from lb to 112.
    # 2. If a solution is found, return it.
    # 3. If time is running out, break and return best found.
    
    # We should allocate time wisely.
    # If we try 105 and it takes 100s, we have no time for 112.
    # So we should set a time limit per scope attempt, or use a global time check.
    # The global time check is in the backtrack.
    
    # To improve performance, we can try 112 first to ensure we have a valid solution?
    # No, the task is to *minimize*. If we find 112, we should still try 105-111 if we have time.
    # But if 105-111 are infeasible, we waste time.
    # Given the complexity, it's likely that 112 is the optimal or close to it.
    # Let's try from lb upwards.
    
    for s in range(lb, 113):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Run search with a local time limit? 
        # The backtrack function uses the global start_time.
        # We can just call it.
        
        success, res = find_solution(s)
        
        if success:
            best_result = res
            best_scope = s
            # Since we are iterating upwards, the first success is the minimum.
            return best_result
        
    # If no solution found in [lb, 112], try to find any solution with a larger scope?
    # Or return the best effort?
    # The task says "Score: the scope... minimized".
    # If we don't find a valid one, we should return a valid one if possible.
    # Let's try a greedy fallback or a wider search.
    
    if best_result is None:
        # Fallback: Try a wider range or a simpler heuristic
        # For now, return a valid solution if we can find one quickly with a larger scope
        # Or just return an empty list? No, must be valid.
        
        # Let's try to find a solution with scope 150 using the same logic but faster?
        # Or just use the greedy method from the original code?
        
        used = set()
        rows = []
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

    return best_result
