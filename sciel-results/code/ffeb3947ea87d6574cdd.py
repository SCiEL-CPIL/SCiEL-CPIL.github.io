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
        next_sq = find_next_square(n, squares, start_time, timeout)
        if next_sq is not None:
            squares.append(next_sq)
        else:
            # If we can't find another one, stop.
            # Note: It's possible that a different choice for the previous square
            # would allow more, but backtracking the entire set is too complex for this time limit.
            break
            
    return squares

def find_next_square(n, existing_squares, start_time, timeout):
    """
    Finds a Latin square orthogonal to all existing_squares using CSP with MRV.
    """
    num_existing = len(existing_squares)
    
    # Pre-compute the values in existing squares at each cell
    existing_vals = [[0] * num_existing for _ in range(n*n)]
    for k, sq in enumerate(existing_squares):
        for i in range(n):
            for j in range(n):
                existing_vals[i*n + j][k] = sq[i][j]

    row_masks = [0] * n
    col_masks = [0] * n
    pair_sets = [set() for _ in range(num_existing)]
    unassigned = set(range(n*n))
    grid = [[0] * n for _ in range(n)]
    
    def is_valid_placement(idx, val):
        i = idx // n
        j = idx % n
        bit = 1 << val
        if (row_masks[i] & bit) or (col_masks[j] & bit):
            return False
        for k in range(num_existing):
            val_k = existing_vals[idx][k]
            pair_key = val * n + val_k
            if pair_key in pair_sets[k]:
                return False
        return True

    def backtrack(depth):
        if time.time() > timeout:
            return False
        if depth == n * n:
            return True
        
        # Select cell with MRV (Minimum Remaining Values)
        best_cell = -1
        best_avail = []
        min_count = n + 1
        
        for idx in unassigned:
            i = idx // n
            j = idx % n
            avail = []
            for val in range(n):
                if is_valid_placement(idx, val):
                    avail.append(val)
            count = len(avail)
            if count == 0:
                return False
            if count < min_count:
                min_count = count
                best_cell = idx
                best_avail = avail
                if count == 1:
                    break
                    
        if best_cell == -1:
            return False
            
        for val in best_avail:
            i = best_cell // n
            j = best_cell % n
            grid[i][j] = val
            bit = 1 << val
            row_masks[i] |= bit
            col_masks[j] |= bit
            for k in range(num_existing):
                val_k = existing_vals[best_cell][k]
                pair_sets[k].add(val * n + val_k)
            unassigned.remove(best_cell)
            
            if backtrack(depth + 1):
                return True
            
            # Undo
            row_masks[i] &= ~bit
            col_masks[j] &= ~bit
            for k in range(num_existing):
                val_k = existing_vals[best_cell][k]
                pair_sets[k].remove(val * n + val_k)
            unassigned.add(best_cell)
            grid[i][j] = 0
            
        return False

    if backtrack(0):
        return grid
    else:
        return None
