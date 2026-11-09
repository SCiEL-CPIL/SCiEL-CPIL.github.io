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
    
    return squares

def find_next_square(n, existing_squares, start_time, timeout):
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    row_masks = [0] * n
    col_masks = [0] * n
    # used_pairs[k][v_new] is a bitmask of v_old values that have been paired with v_new
    # for the k-th existing square
    used_pairs = [[0] * n for _ in existing_squares]
    
    unassigned = [(i, j) for i in range(n) for j in range(n)]
    
    def backtrack():
        if time.time() > timeout:
            return False
            
        if not unassigned:
            return True
        
        min_count = n + 1
        best_cell_idx = -1
        best_domain = 0
        
        for idx, (i, j) in enumerate(unassigned):
            count = 0
            domain_mask = 0
            for v in range(n):
                bit = 1 << v
                if (row_masks[i] & bit) or (col_masks[j] & bit):
                    continue
                
                valid = True
                for k in range(len(existing_squares)):
                    v_k = existing_squares[k][i][j]
                    if used_pairs[k][v] & (1 << v_k):
                        valid = False
                        break
                
                if valid:
                    count += 1
                    domain_mask |= bit
            
            if count == 0:
                return False
            
            if count < min_count:
                min_count = count
                best_cell_idx = idx
                best_domain = domain_mask
                if count == 1:
                    break
        
        i, j = unassigned[best_cell_idx]
        unassigned.pop(best_cell_idx)
        
        v = 0
        while v < n:
            bit = 1 << v
            if best_domain & bit:
                grid[i][j] = v
                row_masks[i] |= bit
                col_masks[j] |= bit
                
                for k in range(len(existing_squares)):
                    v_k = existing_squares[k][i][j]
                    used_pairs[k][v] |= (1 << v_k)
                
                if backtrack():
                    return True
                
                grid[i][j] = -1
                row_masks[i] &= ~bit
                col_masks[j] &= ~bit
                
                for k in range(len(existing_squares)):
                    v_k = existing_squares[k][i][j]
                    used_pairs[k][v] &= ~(1 << v_k)
            
            v += 1
        
        unassigned.insert(best_cell_idx, (i, j))
        return False

    if backtrack():
        return grid
    else:
        return None
