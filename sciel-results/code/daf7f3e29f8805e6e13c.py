import time
import random
from itertools import combinations

def construct(n_rows, k):
    # Fallback greedy
    def greedy_construct():
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

    start_time = time.time()
    time_limit = 110.0
    
    # Lower bound
    diffs_per_row = (k + 1) * k // 2
    lb = n_rows * diffs_per_row
    
    # First, get a quick upper bound using greedy to limit the search space
    # Greedy usually provides a valid, albeit suboptimal, M
    initial_rows = greedy_construct()
    M_upper = max(max(row) for row in initial_rows)
    
    # Search M from lb up to M_upper
    # We want the smallest M.
    # Randomized DFS with restarts is better than deterministic DFS for finding ANY solution.
    
    def randomized_dfs(M, max_time_per_try=5.0):
        used = set()
        rows = []
        attempts = 0
        start_dfs_time = time.time()
        
        # We will try multiple random restarts
        while time.time() - start_time < time_limit - 5:
            # Reset state for a new attempt
            used = set()
            rows = []
            attempt_start = time.time()
            
            # Use a different random seed for each attempt to explore different parts of the tree
            random.seed(int(attempt_start * 1000) % (2**32))
            
            def dfs(row_idx, curr_row):
                if time.time() - attempt_start > max_time_per_try:
                    return False
                
                if len(curr_row) == k + 1:
                    if row_idx == n_rows - 1:
                        rows.append(list(curr_row))
                        return True
                    return dfs(row_idx + 1, [0])
                
                last = curr_row[-1]
                start_val = last + 1
                if not curr_row:
                    start_val = 1
                
                # Generate candidates
                candidates = list(range(start_val, M + 1))
                
                # Heuristic: Sort by "cost" or just shuffle?
                # Shuffling is a strong heuristic for escaping bad local minima in the search tree.
                # However, for minimization, smaller values are generally better.
                # Let's use a biased shuffle: smaller values have higher probability of being tried first?
                # Or just full shuffle to ensure we don't get stuck in a "lexicographic trap".
                random.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = {x - a for a in curr_row}
                    
                    if new_diffs & used:
                        continue
                    
                    # Check if we can possibly complete the row
                    # We need (k + 1 - len(curr_row) - 1) more elements after x.
                    # The max value is M.
                    # This check is weak, so we skip complex lookahead for speed.
                    
                    used.update(new_diffs)
                    curr_row.append(x)
                    
                    if dfs(row_idx, curr_row):
                        return True
                    
                    curr_row.pop()
                    used.difference_update(new_diffs)
                
                return False

            if dfs(0, [0]):
                return rows
            attempts += 1
            
    # Iterate M from lower bound to upper bound
    # We can skip M values if they are too tight? No, we must check all.
    # But we can stop if we find a solution.
    
    # Optimization: The lower bound is often not tight.
    # Start from lb, but if lb is too low, DFS will fail quickly.
    
    for M in range(lb, M_upper + 1):
        if time.time() - start_time > time_limit - 10:
            break
            
        result = randomized_dfs(M)
        if result:
            return result

    # If no solution found within time/M limits, return greedy result
    return initial_rows
