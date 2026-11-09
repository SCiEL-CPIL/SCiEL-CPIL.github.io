import time

def construct():
    n = 10
    # L1: Cyclic Latin Square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1]
    
    # Try to find one more square using backtracking
    new_square = find_orthogonal_square_backtrack(squares, n)
    if new_square:
        squares.append(new_square)
        
    return squares

def find_orthogonal_square_backtrack(fixed_squares, n):
    # Grid to fill
    grid = [[None] * n for _ in range(n)]
    
    # Constraints for Latin Square
    # row_symbols[i][v] = True if v is in row i
    # col_symbols[j][v] = True if v is in col j
    row_symbols = [set() for _ in range(n)]
    col_symbols = [set() for _ in range(n)]
    
    # Constraints for Orthogonality
    # For each fixed square L_fix, pair_counts[(L_fix[i][j], grid[i][j])] must be unique
    # We can maintain a set of used pairs for each fixed square
    used_pairs = [set() for _ in fixed_squares]
    
    # Precompute fixed square values for quick access
    fixed_vals = [[L[i][j] for L in fixed_squares] for i in range(n) for j in range(n)]
    # Actually, let's just index fixed_squares[k][i][j]
    
    # Order cells: MRV (Minimum Remaining Values) heuristic
    # Initially all cells have 10 values.
    # We will use a simple recursive backtracking with forward checking.
    
    # To speed up, we can fill row by row, but MRV is better.
    # For n=10, simple row-by-row with forward checking might be sufficient if we have good pruning.
    
    def is_valid(i, j, val):
        # Check Latin constraints
        if val in row_symbols[i]:
            return False
        if val in col_symbols[j]:
            return False
        
        # Check Orthogonality constraints
        for k, L_fix in enumerate(fixed_squares):
            pair = (L_fix[i][j], val)
            if pair in used_pairs[k]:
                return False
        
        return True

    def solve(idx):
        if idx == n * n:
            return True
            
        i = idx // n
        j = idx % n
        
        # Try all values
        # Optimization: Use MRV? For simplicity, try 0..n-1
        # But we can check which values are still possible
        for val in range(n):
            if is_valid(i, j, val):
                # Place
                grid[i][j] = val
                row_symbols[i].add(val)
                col_symbols[j].add(val)
                for k, L_fix in enumerate(fixed_squares):
                    used_pairs[k].add((L_fix[i][j], val))
                
                if solve(idx + 1):
                    return True
                
                # Unplace
                grid[i][j] = None
                row_symbols[i].remove(val)
                col_symbols[j].remove(val)
                for k, L_fix in enumerate(fixed_squares):
                    used_pairs[k].remove((L_fix[i][j], val))
                    
        return False

    # Start solving
    # To make it faster, we can fix the first row to 0..n-1 (isomorphic to cyclic)
    # This reduces symmetry.
    for j in range(n):
        grid[0][j] = j
        row_symbols[0].add(j)
        col_symbols[j].add(j)
        for k, L_fix in enumerate(fixed_squares):
            used_pairs[k].add((L_fix[0][j], j))
            
    if solve(1 * n): # Start from row 1
        return grid
    else:
        return None
