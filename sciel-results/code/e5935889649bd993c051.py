import time
import random
import sys

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows and k+1 entries per row.
    Strategy: Backtracking search with bitmask difference tracking.
    We build rows one by one. For each row, we fill entries from left to right.
    The bitmask tracks all differences used so far.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    m = k + 1  # entries per row
    # Max possible difference estimate for bitmask size.
    # Based on task description, best known is 134. We'll allow up to 255 to be safe.
    MAX_DIFF = 255
    MASK_ALL = (1 << (MAX_DIFF + 1)) - 1  # Bits 0 to MAX_DIFF
    
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    best_solution = None
    best_scope = float('inf')
    
    # Precompute powers of 2 for differences
    pow2 = [1 << i for i in range(MAX_DIFF + 1)]
    
    def backtrack(row_idx, row, used_mask, current_scope):
        """
        Recursively fill row_idx.
        row: current list of values for this row (starts with 0)
        used_mask: integer bitmask of all differences used in all previous rows AND current row so far.
        """
        nonlocal best_solution, best_scope
        
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            # All rows filled
            scope = current_scope
            if scope < best_scope:
                best_scope = scope
                best_solution = [list(r) for r in rows]
            return True
            
        # If we are filling a new row, start with [0]
        if not row:
            row = [0]
            
        if len(row) == m:
            # Row complete, move to next
            return backtrack(row_idx + 1, [], used_mask, current_scope)
        
        # Determine the next value to add
        prev_val = row[-1]
        
        # Heuristic: Start from the smallest possible value that doesn't conflict.
        # We can also randomize the start point slightly to explore different branches.
        # To minimize scope, we generally want small values.
        
        # Lower bound: prev_val + 1
        # Upper bound: We want to keep scope small. 
        # If we have a current best_scope, we shouldn't exceed it (unless we are improving).
        # Actually, we can exceed current best_scope if we haven't finished, 
        # but if current_scope (max of previous rows) is already >= best_scope, 
        # we can prune? No, because we might reduce previous rows? No, rows are fixed once done.
        # So if current_scope >= best_scope, we can prune immediately.
        if current_scope >= best_scope:
            return False
            
        # Try values from low to high
        # Optimization: We can skip values that create duplicate diffs within the current row immediately.
        
        # To add variety and avoid getting stuck in local minima of the greedy "smallest first" path,
        # we can occasionally try a random jump or start from a slightly higher value.
        # However, strict "smallest first" is usually best for minimizing max value in CSP.
        
        # Let's try a range.
        # Max value we can try: 
        # If we are on the last row, we can try up to best_scope - 1.
        # Otherwise, we can try up to best_scope - 1 as well, because if we exceed it, 
        # the final scope will be > best_scope.
        
        max_val_to_try = best_scope - 1
        if max_val_to_try < prev_val + 1:
            return False
            
        # Randomization: 10% of the time, start from a random value in the upper half
        # to explore wider structures, but usually stick to greedy.
        start_val = prev_val + 1
        if rng.random() < 0.1:
             start_val = rng.randint(prev_val + 1, max_val_to_try)
             
        # We iterate from start_val to max_val_to_try
        # But to be efficient, we should check the bitmask quickly.
        
        for val in range(start_val, max_val_to_try + 1):
            if time.time() > deadline:
                return False
                
            # Check if adding 'val' creates any conflict with existing differences
            # Differences to check: val - row[i] for all i in row
            # We need to ensure:
            # 1. No diff is in used_mask
            # 2. No two new diffs are identical to each other (within this row step)
            
            new_diffs = []
            conflict = False
            mask_to_add = 0
            
            for i in range(len(row)):
                d = val - row[i]
                if d > MAX_DIFF:
                    conflict = True
                    break
                bit = pow2[d]
                if used_mask & bit:
                    conflict = True
                    break
                if mask_to_add & bit:
                    # Duplicate diff within the new set
                    conflict = True
                    break
                mask_to_add |= bit
                new_diffs.append(d)
            
            if conflict:
                continue
                
            # Valid move
            row.append(val)
            new_scope = max(current_scope, val)
            
            # Pruning: If new_scope >= best_scope, and we haven't finished the row,
            # we might still be able to finish with a smaller max? No, val is the max so far.
            # If val >= best_scope, the final scope will be >= best_scope.
            # So we can prune here if val >= best_scope.
            # Wait, best_scope is the global best found so far.
            # If val >= best_scope, this branch cannot improve on best_scope.
            # So we can skip this val entirely?
            # Yes, because we want to minimize the max entry.
            if val >= best_scope:
                row.pop()
                continue
            
            # Recurse
            if backtrack(row_idx, row, used_mask | mask_to_add, new_scope):
                return True
                
            row.pop()
            
        return False

    # To improve the initial bound, run a few greedy seeds first
    def greedy_seed():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows

    # Initialize best_solution with a greedy seed to have a finite upper bound
    # This is crucial for the backtracking to prune effectively.
    seed_rows = greedy_seed()
    if seed_rows:
        # Validate seed
        used_check = set()
        valid = True
        scope = 0
        for row in seed_rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_check:
                        valid = False
                    used_check.add(d)
            if row[-1] > scope:
                scope = row[-1]
        if valid:
            best_solution = [list(r) for r in seed_rows]
            best_scope = scope
        else:
            best_scope = 255 # Fallback large bound
            best_solution = None
            
    # Run backtracking
    # We pass an empty list for the first row to trigger the [0] initialization
    backtrack(0, [], 0, 0)
    
    if best_solution is None:
        # Fallback if backtracking failed completely (shouldn't happen with good bound)
        return seed_rows if seed_rows else [[0]*m for _ in range(n_rows)]
        
    return best_solution

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
