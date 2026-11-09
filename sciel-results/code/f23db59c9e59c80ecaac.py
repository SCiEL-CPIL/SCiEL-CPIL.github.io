import random
import time

def construct():
    n = 10
    start_time = time.time()
    timeout = 55.0 # Leave some buffer
    
    # Generate first square (cyclic)
    squares = []
    sq0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares.append(sq0)
    
    # Try to find more squares
    for _ in range(8): # Max 9 total
        if time.time() - start_time > timeout:
            break
        
        # Backtracking to find next square
        next_sq = find_next_square(n, squares)
        if next_sq is not None:
            squares.append(next_sq)
    
    return squares

def find_next_square(n, existing_squares):
    # Initialize grid
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Bitmasks for rows and cols
    row_masks = [0] * n
    col_masks = [0] * n
    
    # Sets of used pairs for each existing square
    # pair_sets[k] is a set of tuples (val_in_new_sq, val_in_existing_sq_k)
    pair_sets = [set() for _ in existing_squares]
    
    # Pre-compute candidates for each cell?
    # Or do it on the fly.
    
    # Order of cells: row-major
    cells = [(i, j) for i in range(n) for j in range(n)]
    
    def backtrack(idx):
        if time.time() > 55.0:
            return False
        
        if idx == n * n:
            return True
        
        i, j = cells[idx]
        
        # Try each symbol 0..9
        # Heuristic: Try symbols that are most constrained?
        # For now, just iterate 0..9
        
        for val in range(n):
            # Check row/col uniqueness
            bit = 1 << val
            if (row_masks[i] & bit) or (col_masks[j] & bit):
                continue
            
            # Check orthogonality
            valid = True
            for k, sq_k in enumerate(existing_squares):
                val_k = sq_k[i][j]
                pair = (val, val_k)
                if pair in pair_sets[k]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Place value
            grid[i][j] = val
            row_masks[i] |= bit
            col_masks[j] |= bit
            
            for k, sq_k in enumerate(existing_squares):
                val_k = sq_k[i][j]
                pair_sets[k].add((val, val_k))
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            grid[i][j] = -1
            row_masks[i] &= ~bit
            col_masks[j] &= ~bit
            for k, sq_k in enumerate(existing_squares):
                val_k = sq_k[i][j]
                pair_sets[k].remove((val, val_k))
        
        return False
    
    if backtrack(0):
        return grid
    else:
        return None
