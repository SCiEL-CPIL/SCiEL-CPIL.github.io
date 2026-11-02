import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 145.0  # 5s margin for safety
    
    # Theoretical lower bound: n_rows * (k * (k+1) / 2)
    # For n_rows=4, k=7: 4 * 28 = 112
    min_scope = n_rows * (k * (k + 1) // 2)
    
    def generate_row(max_scope):
        """
        Generates a random row of k+1 integers starting with 0, strictly increasing,
        with max entry <= max_scope, and unique internal differences.
        Returns None if not possible.
        """
        row = [0]
        used_diffs = set()
        
        # We need to pick k more numbers.
        # Greedy with randomization.
        # To ensure we don't get stuck, we can look ahead or use backtracking, 
        # but for small k, simple greedy with random restarts is usually okay.
        # However, to maximize diversity, we introduce noise.
        
        for _ in range(k):
            last = row[-1]
            # Next number must be > last and <= max_scope
            # Also, differences with all previous elements must be new.
            
            # Find all valid candidates
            valids = []
            x = last + 1
            while x <= max_scope:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valids.append(x)
                x += 1
            
            if not valids:
                return None
            
            # Choose a candidate
            # Bias towards smaller values to keep scope low?
            # But we are constrained by max_scope anyway.
            # Random choice ensures diversity.
            # Maybe weighted choice: prefer smaller to leave room for future steps?
            # If we pick too large, we might run out of space for remaining k-1 steps.
            
            # Heuristic: The remaining steps need at least (k - len(row)) + 1 space?
            # Actually, just picking randomly from valids is a good start.
            # Let's pick from the first 20% or so to keep it compact?
            # Or just random.
            
            # Let's try a mix: 50% random, 50% smallest valid.
            if random.random() < 0.5:
                choice = valids[0]
            else:
                choice = random.choice(valids)
            
            row.append(choice)
            for a in row[:-1]:
                used_diffs.add(choice - a)
        
        return row

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i):
                diffs.add(row[i] - row[j])
        return diffs

    def find_independent_set(candidates, n):
        """
        Finds n candidates from the list that have disjoint difference sets.
        candidates: list of (scope, row, diffs)
        Returns list of rows or None.
        """
        if not candidates:
            return None
        
        # Sort by scope to try smaller ones first? 
        # Or by "popularity" (number of conflicts)?
        # Sorting by scope is a good heuristic for minimizing max scope.
        # But we already filtered by max_scope in the outer loop.
        
        # Backtracking search
        # Since n is small (4), we can use a simple recursive function.
        
        def backtrack(idx, selected, used_diffs_global):
            if len(selected) == n:
                return selected[:]
            
            # Pruning: if not enough candidates left
            if idx + (n - len(selected)) > len(candidates):
                return None
            
            for i in range(idx, len(candidates)):
                scope, row, diffs = candidates[i]
                
                # Check conflict
                if diffs & used_diffs_global:
                    continue
                
                # Select
                selected.append(row)
                new_used = used_diffs_global | diffs
                
                res = backtrack(i + 1, selected, new_used)
                if res:
                    return res
                
                # Backtrack
                selected.pop()
            
            return None

        return backtrack(0, [], set())

    best_solution = None
    
    # We will iterate on the scope limit
    # Start from theoretical minimum
    current_scope = min_scope
    
    # To avoid infinite loops on impossible small scopes, 
    # we can check if generation yields any rows.
    
    while time.time() < deadline:
        # Generate a pool of candidates for this scope
        pool = []
        # How many to generate?
        # Let's start with 1000 and increase if needed?
        # Or just generate as many as possible in a time slice.
        
        gen_start = time.time()
        target_pool_size = 2000
        attempts = 0
        max_attempts = 10000 # Safety cap on attempts per scope
        
        while len(pool) < target_pool_size and attempts < max_attempts and time.time() < deadline - 5:
            row = generate_row(current_scope)
            if row:
                # Check if this row is already in the pool (to avoid duplicates)
                # Using a set of tuples for rows
                row_tuple = tuple(row)
                if row_tuple not in [tuple(r[1]) for r in pool]: # O(N) check, might be slow
                    pass # We'll just allow duplicates, it's okay for MIS search, just less efficient.
                
                diffs = get_diffs(row)
                pool.append((row[-1], row, diffs))
            attempts += 1
        
        if not pool:
            current_scope += 1
            continue
            
        # Remove duplicates to save search time
        unique_pool = []
        seen = set()
        for scope, row, diffs in pool:
            rt = tuple(row)
            if rt not in seen:
                seen.add(rt)
                unique_pool.append((scope, row, diffs))
        
        # Sort by scope
        unique_pool.sort(key=lambda x: x[0])
        
        # Try to find independent set
        # We can limit the search time here
        search_start = time.time()
        
        # To speed up, we can use a greedy heuristic first?
        # Or just run the backtracker.
        
        # Optimization: If the pool is large, the backtracker might be slow.
        # We can limit the number of candidates we pass to the backtracker.
        # Since we want to minimize scope, and we are iterating scope upwards,
        # all candidates have scope <= current_scope.
        # We just need *any* independent set of size n_rows.
        
        # Let's try the backtracker on the whole pool.
        # If it's too slow, we can truncate the pool to the first 1000 candidates.
        
        candidates_for_search = unique_pool[:3000] # Limit size
        
        res = find_independent_set(candidates_for_search, n_rows)
        
        if res:
            return res
        
        # If not found, increase scope
        current_scope += 1
        
        # If we've been searching for a long time without success, 
        # maybe we should increase scope faster?
        # Or generate more candidates?
        
    # Fallback if no solution found (should not happen for reasonable inputs)
    # Return a greedy solution (might not be optimal scope, but valid)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# Test/Run guard
if __name__ == "__main__":
    # Example call
    res = construct(4, 7)
    print(res)
