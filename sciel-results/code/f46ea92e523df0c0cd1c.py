import time
import random
import sys

def construct(n_rows, k):
    row_len = k + 1
    total_diffs_per_row = row_len * (row_len - 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    lb = total_diffs
    
    best_rows = None
    best_scope = float('inf')
    
    deadline = time.time() + 110
    
    # Try scopes from lb upwards
    # We'll give each scope a time budget
    # If we find a solution, we stop.
    
    # To handle potential slowness, we'll use a randomized backtracker
    # that tries to find a solution for a given M.
    
    for M in range(lb, lb + 30):
        if time.time() > deadline:
            break
            
        # Try to find a solution with scope M
        # We'll try a few random seeds for the value ordering
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            sol = _solve_for_scope_randomized(n_rows, row_len, M, deadline, random.Random(attempt + M))
            if sol is not None:
                best_rows = sol
                best_scope = M
                found = True
                break
                
        if found:
            break
            
    if best_rows is None:
        # Fallback
        best_rows = _greedy_construct(n_rows, k)
        
    return best_rows

def _solve_for_scope_randomized(n_rows, row_len, M, deadline, rng):
    used_diffs = [False] * (M + 1)
    rows = []
    
    # Pre-compute nothing, just use the set logic
    # To speed up, we can use a list of available diffs?
    # M is small (105-135), so boolean array is fast.
    
    def backtrack(row_idx, current_row, min_max):
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
            
        # Symmetry: rows must be sorted by max element
        # min_max is the max of the previous row
        
        # Heuristic: Try to fill the row
        # We'll shuffle the candidate values?
        # No, we need to ensure increasing order in the row.
        # We can randomize the *selection* strategy.
        
        # Standard backtracking with pruning
        def fill_row(idx, last_val):
            if idx == row_len:
                if current_row[-1] < min_max:
                    return False
                rows.append(current_row[:])
                if backtrack(row_idx + 1, [0], current_row[-1]):
                    return True
                rows.pop()
                return False
                
            # Generate candidates
            # To introduce randomness, we can pick a random valid candidate
            # from a pool of the first few valid ones?
            # Or just iterate 1..M.
            
            # Optimization: 
            # If we are at the last element (idx == row_len - 1), 
            # we just need to find any valid v.
            
            candidates = []
            for v in range(last_val + 1, M + 1):
                valid = True
                new_diffs = []
                for i in range(len(current_row)):
                    d = v - current_row[i]
                    if d > M:
                        valid = False
                        break
                    if used_diffs[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((v, new_diffs))
                    
            if not candidates:
                return False
            
            # Randomize order of candidates?
            # If we randomize, we might find a solution faster in some branches.
            # But it might also make it slower.
            # Let's try a mix: 50% ordered, 50% shuffled.
            if rng.random() < 0.5:
                rng.shuffle(candidates)
                
            for v, new_diffs in candidates:
                current_row.append(v)
                for d in new_diffs:
                    used_diffs[d] = True
                    
                if fill_row(idx + 1, v):
                    return True
                    
                current_row.pop()
                for d in new_diffs:
                    used_diffs[d] = False
                    
            return False
            
        return fill_row(1, 0)
        
    current_row = [0]
    return backtrack(0, current_row, 0)

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
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
