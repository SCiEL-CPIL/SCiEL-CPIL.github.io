import random
import time

def construct():
    n = 10
    # L1: Cyclic Latin Square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1]
    
    # Try to find more squares
    # We will try to find up to 5 more squares (total 6)
    # Time limit per square search: 10 seconds
    # Total time limit: 50 seconds (leave 10 for overhead)
    
    start_time = time.time()
    
    for _ in range(5):
        if time.time() - start_time > 50:
            break
            
        new_square = find_orthogonal_square(squares, n, timeout=10)
        if new_square:
            squares.append(new_square)
        else:
            # If we can't find one, break
            # We could try restarting with different random seeds, 
            # but if local search fails, it might be hard.
            break
            
    return squares

def find_orthogonal_square(fixed_squares, n, timeout=10):
    # Generate a random Latin Square
    L = generate_random_latin_square(n)
    
    start = time.time()
    # Local Search
    # Objective: Minimize conflicts with all fixed_squares
    
    # Precompute conflicts
    # We will use a simple hill climbing with random restarts if stuck
    
    best_L = L
    best_conflicts = count_conflicts(best_L, fixed_squares, n)
    
    while time.time() - start < timeout:
        if best_conflicts == 0:
            return best_L
            
        # Pick a random cell
        r = random.randint(0, n - 1)
        c = random.randint(0, n - 1)
        
        # Try all possible values for L[r][c]
        # But we must maintain Latin property?
        # Standard Min-Conflicts for Latin Squares:
        # We can swap values in a row or column to maintain Latin property.
        # Or, we can just change the value and accept violations, 
        # but then we need to repair Latin property too.
        
        # Simpler: 
        # 1. Pick a row.
        # 2. Pick two columns in that row.
        # 3. Swap the values.
        # This maintains the Latin property for that row.
        # It might break the column property.
        # So we need to track both row and column conflicts.
        
        # Let's define the objective as:
        # OrthoConflicts + RowConflicts + ColConflicts
        
        current_conflicts = best_conflicts
        # We need to calculate the delta.
        
        # For efficiency, we just evaluate the new state.
        
        # Strategy:
        # 1. Pick a random row r.
        # 2. Pick two random columns c1, c2.
        # 3. Swap L[r][c1] and L[r][c2].
        # 4. Calculate new conflicts.
        # 5. If better, accept. If equal, accept with prob p.
        # 6. Else, revert.
        
        # We also need to ensure we don't get stuck.
        # If no improvement for X iterations, restart with new random square.
        
        no_improve_count = 0
        max_no_improve = 1000
        
        L_curr = [row[:] for row in best_L]
        curr_conflicts = best_conflicts
        
        while time.time() - start < timeout:
            if curr_conflicts == 0:
                return L_curr
                
            r = random.randint(0, n - 1)
            c1 = random.randint(0, n - 1)
            c2 = random.randint(0, n - 1)
            if c1 == c2:
                continue
                
            # Swap
            L_curr[r][c1], L_curr[r][c2] = L_curr[r][c2], L_curr[r][c1]
            
            new_conflicts = count_conflicts(L_curr, fixed_squares, n)
            
            if new_conflicts <= curr_conflicts:
                curr_conflicts = new_conflicts
                no_improve_count = 0
                if curr_conflicts < best_conflicts:
                    best_conflicts = curr_conflicts
                    best_L = [row[:] for row in L_curr]
            else:
                # Revert
                L_curr[r][c1], L_curr[r][c2] = L_curr[r][c2], L_curr[r][c1]
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                # Restart
                L_curr = generate_random_latin_square(n)
                curr_conflicts = count_conflicts(L_curr, fixed_squares, n)
                if curr_conflicts < best_conflicts:
                    best_conflicts = curr_conflicts
                    best_L = [row[:] for row in L_curr]
                no_improve_count = 0
                max_no_improve += 1000 # Increase patience slightly? Or keep same.
                
    return None

def count_conflicts(L, fixed_squares, n):
    # Count orthogonality conflicts
    # And Latin property conflicts (rows/cols)
    # But L is generated as Latin, and swaps preserve row Latin.
    # Swaps might break column Latin.
    # So we must check column Latin property.
    
    conflicts = 0
    
    # Check Column Latin Property
    for c in range(n):
        col = [L[r][c] for r in range(n)]
        if len(set(col)) != n:
            # Count duplicates
            seen = {}
            for val in col:
                seen[val] = seen.get(val, 0) + 1
            for count in seen.values():
                if count > 1:
                    conflicts += count - 1
                    
    # Check Orthogonality
    for L_fix in fixed_squares:
        pairs = {}
        for i in range(n):
            for j in range(n):
                pair = (L_fix[i][j], L[i][j])
                pairs[pair] = pairs.get(pair, 0) + 1
        for count in pairs.values():
            if count > 1:
                conflicts += count - 1
                
    return conflicts

def generate_random_latin_square(n):
    # Simple backtracking
    grid = [[None] * n for _ in range(n)]
    
    def is_valid(i, j, val):
        for k in range(n):
            if grid[i][k] == val or grid[k][j] == val:
                return False
        return True
        
    def solve(i, j):
        if i == n:
            return True
        if j == n:
            return solve(i + 1, 0)
            
        if grid[i][j] is not None:
            return solve(i, j + 1)
            
        vals = list(range(n))
        random.shuffle(vals)
        for val in vals:
            if is_valid(i, j, val):
                grid[i][j] = val
                if solve(i, j + 1):
                    return True
                grid[i][j] = None
        return False
        
    # To make it faster, we can fill row by row
    # But for n=10, this is fine.
    # We need to ensure it's fast.
    # Backtracking for n=10 can be slow if unlucky.
    # Let's use a heuristic: fill the first row with 0..n-1.
    for j in range(n):
        grid[0][j] = j
        
    # Then solve the rest
    if solve(1, 0):
        return grid
    else:
        # Fallback to cyclic
        return [[(i + j) % n for j in range(n)] for i in range(n)]
