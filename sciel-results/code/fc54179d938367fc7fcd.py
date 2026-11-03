import time
import random
from itertools import combinations

def construct(n_rows, k):
    # n_rows = 4, k = 7 (so 8 marks)
    # Deadline
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate candidate rows (Golomb rulers) of length k+1
    # We want to minimize the max entry.
    # We'll generate all valid Golomb rulers with max entry up to a limit.
    # The best known is 134. Let's search up to 140 to be safe, or dynamically.
    
    max_scope_limit = 150
    
    candidates = []
    
    # Helper to generate Golomb rulers
    def generate_rulers(max_val):
        rulers = []
        used_diffs = set()
        current = [0]
        
        def backtrack(start_idx, current_marks, current_diffs):
            if time.time() > deadline:
                return
            if len(current_marks) == k + 1:
                # Found a complete ruler
                rulers.append((current_marks.copy(), current_diffs.copy()))
                return
            
            last = current_marks[-1]
            # Next mark must be > last
            # Also, to prune, if we are at the last position, we can bound it?
            # For now, try all x > last up to max_val
            
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in current_marks}
                if new_diffs & current_diffs:
                    continue
                if new_diffs & used_diffs_global: # This is for the global set in packing, not here
                    pass # Not applicable here, this is for single ruler generation
                
                # Check internal distinctness
                # new_diffs are distinct from each other? 
                # Since x > a_i > a_j, x-a_i < x-a_j. 
                # Also x-a_i != a_j - a_i (obviously).
                # We just need new_diffs disjoint from current_diffs.
                
                current_marks.append(x)
                current_diffs |= new_diffs
                backtrack(start_idx + 1, current_marks, current_diffs)
                current_marks.pop()
                current_diffs -= new_diffs
                
                # Pruning: if current max is already too high, we could stop? 
                # No, we are generating all up to max_val.
        
        backtrack(0, current, set())
        return rulers

    # The above generation might be slow if max_val is high.
    # Let's optimize generation.
    # We know the max entry of the best solution is 134.
    # We can generate rulers up to 135 first. If we find a solution, great.
    # If not, increase limit.
    
    best_solution = None
    best_max = float('inf')
    
    # We will iterate through possible max scopes
    for limit in range(112, 150): # Start from counting bound
        if time.time() > deadline:
            break
            
        # Generate all Golomb rulers with max mark <= limit
        # To speed up, we can cache rulers found in previous limits? 
        # A ruler with max mark L is also a valid candidate for limit L+1.
        
        # Let's just generate up to 'limit' each time, or keep a global list and filter.
        # Better: Generate once up to 150, then filter by max mark.
        
    # Let's generate all rulers up to 150 once.
    # This might take time. Let's estimate.
    # Number of Golomb rulers of length 8 is not huge.
    
    all_rulers = []
    # We'll use a more efficient generator if possible, or just the backtracking above.
    # The backtracking above explores x from last+1 to max_val.
    # This is fine for max_val=150.
    
    def gen_rulers(max_val):
        res = []
        curr = [0]
        diffs = set()
        
        def rec():
            if time.time() > deadline:
                return
            if len(curr) == k + 1:
                res.append((tuple(curr), frozenset(diffs)))
                return
            
            last = curr[-1]
            # Heuristic: try smaller x first? Doesn't matter for generation.
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in curr}
                if new_diffs & diffs:
                    continue
                curr.append(x)
                diffs |= new_diffs
                rec()
                curr.pop()
                diffs -= new_diffs
        
        rec()
        return res

    all_rulers = gen_rulers(150)
    
    # Sort rulers by max mark to try smaller scopes first
    all_rulers.sort(key=lambda r: r[0][-1])
    
    # Now, find 4 rulers with disjoint difference sets
    # Backtracking search
    
    found = None
    
    def find_packing(rulers, num_rows, used_diffs, current_rows, current_max):
        nonlocal found, best_max
        
        if time.time() > deadline:
            return
        
        if num_rows == 0:
            if current_max < best_max:
                best_max = current_max
                found = [list(r[0]) for r in current_rows]
            return
        
        # Prune: if current_max >= best_max, we can't improve
        # Note: current_max is the max of the rows chosen so far.
        # The new row might be smaller, but the max won't decrease.
        # So if current_max >= best_max, we can prune? 
        # Yes, because we are trying to minimize the global max.
        
        if current_max >= best_max:
            return
            
        # Try each ruler
        # Optimization: only consider rulers with max mark <= best_max
        for i in range(len(rulers)):
            ruler_marks, ruler_diffs = rulers[i]
            r_max = ruler_marks[-1]
            
            if r_max >= best_max:
                # Since sorted, we can break? No, because a later ruler might be smaller? 
                # No, sorted by max mark. So if this one is >= best_max, all subsequent are too.
                break
                
            new_max = max(current_max, r_max)
            if new_max >= best_max:
                break
                
            if ruler_diffs & used_diffs:
                continue
                
            # Choose this ruler
            current_rows.append(ruler)
            # Note: used_diffs is a set. We need to add/remove.
            # frozenset doesn't support |=. Let's use set for used_diffs in recursion.
            
            # To avoid modifying the set in place, we pass a new set or use a global stack.
            # Given small depth (4), creating new sets is fine.
            new_used = used_diffs | ruler_diffs
            
            find_packing(rulers[i+1:], num_rows - 1, new_used, current_rows, new_max)
            
            current_rows.pop()
            
            # Optional: Pruning based on remaining rows?
            # If we have chosen i rows, and we need 4, can we bound the max?
            # Not easily without more info.
            
    # Initial call
    # We want to pick 4 distinct rulers.
    # The list `all_rulers` contains unique rulers? 
    # Yes, each tuple of marks is unique.
    
    # We need to ensure we pick 4 *different* rulers.
    # The recursive call `find_packing(rulers[i+1:], ...)` ensures we pick distinct rulers from the list.
    # However, we might have duplicate rulers in `all_rulers`? No, generated uniquely.
    
    # One issue: `all_rulers` might contain the same ruler multiple times if generated differently?
    # No, the backtracking generates each combination of marks once.
    
    find_packing(all_rulers, n_rows, set(), [], 0)
    
    if found:
        return found
    else:
        # Fallback to greedy if no solution found within time/limits
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
