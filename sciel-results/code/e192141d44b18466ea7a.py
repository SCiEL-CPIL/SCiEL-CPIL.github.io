import time
import sys

def construct():
    n = 10
    start_time = time.time()
    timeout = 55.0
    
    # Square 0: Cyclic
    sq0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [sq0]
    
    # Try to find up to 8 more squares (total 9? Max MOLS for n=10 is 8, so total 9 squares? 
    # Wait, MOLS count is usually k squares. If we have k MOLS, we return k squares.
    # The problem asks for "list of 10-by-10 arrays". 
    # "Maximize the number of pairwise orthogonal arrays."
    # Known max for n=10 is 8 MOLS. So we can return 8 squares.
    # The seed returns 1.
    # I will try to find 8 squares total.
    
    for _ in range(8):
        if time.time() - start_time > timeout:
            break
            
        next_sq = find_next_square_csp(n, squares, start_time, timeout)
        if next_sq is not None:
            squares.append(next_sq)
        else:
            # If we can't find one, maybe try a different randomization or break?
            # Given the difficulty, if the first few fail, later ones likely will too or take forever.
            # But for n=10, it is known that 8 exist. A good solver should find them.
            break
            
    return squares

def find_next_square_csp(n, existing_squares, start_time, timeout):
    # Pre-compute for each existing square, for each value v, the list of positions (i,j) where sq[i][j] == v
    # This helps in propagation: if we assign v to (i,j), and sq_k[i][j] = u, 
    # then for all (i',j') in pos[sq_k][u], we cannot assign v.
    
    pos_in_sq = []
    for sq in existing_squares:
        pos = {v: [] for v in range(n)}
        for i in range(n):
            for j in range(n):
                pos[sq[i][j]].append((i, j))
        pos_in_sq.append(pos)
        
    # Grid and domains
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    domains = [[set(range(n)) for _ in range(n)] for _ in range(n)]
    
    # Symmetry Breaking: Fix first row to 0..9
    # This is valid because we can permute symbols in a Latin square.
    for j in range(n):
        grid[0][j] = j
        domains[0][j] = {j}
        
    # Data structures for constraints
    row_masks = [0] * n
    col_masks = [0] * n
    # pair_used[k] is a set of (val_new, val_existing_k)
    pair_used = [set() for _ in existing_squares]
    
    # Initialize with first row assignments
    for j in range(n):
        val = j
        bit = 1 << val
        row_masks[0] |= bit
        col_masks[j] |= bit
        
        for k, sq_k in enumerate(existing_squares):
            val_k = sq_k[0][j]
            pair_used[k].add((val, val_k))
            
            # Propagate: For all (i', j') in pos_in_sq[k][val_k], remove val from domains[i'][j']
            # Note: (0,j) is already assigned, so skip if i==0 and j==j? 
            # Actually, we should remove from all unassigned cells.
            for (i2, j2) in pos_in_sq[k][val_k]:
                if i2 == 0 and j2 == j:
                    continue
                if val in domains[i2][j2]:
                    domains[i2][j2].remove(val)
                    
    def is_timeout():
        return time.time() - start_time > timeout

    def backtrack():
        if is_timeout():
            return False
            
        # Find unassigned cell with smallest domain (MRV)
        min_size = n + 1
        min_i = -1
        min_j = -1
        for i in range(n):
            for j in range(n):
                if grid[i][j] == -1:
                    size = len(domains[i][j])
                    if size == 0:
                        return False # Contradiction
                    if size < min_size:
                        min_size = size
                        min_i = i
                        min_j = j
                        if size == 1:
                            break
            if min_size == 1:
                break
                
        if min_i == -1:
            return True # All assigned
            
        i, j = min_i, min_j
        # Try values in domain
        # Heuristic: Try values that are "least constraining"? 
        # For now, iterate through sorted domain for determinism or random?
        # Random might help escape local minima, but MRV is strong.
        for val in list(domains[i][j]):
            # Check if val is still valid (it should be, as it's in domain)
            # But we need to check if it creates a conflict in row/col? 
            # The domain should have been pruned. 
            # However, row_masks/col_masks are maintained.
            bit = 1 << val
            if (row_masks[i] & bit) or (col_masks[j] & bit):
                continue # Should not happen if domains are consistent, but safety check
            
            # Tentative assignment
            grid[i][j] = val
            row_masks[i] |= bit
            col_masks[j] |= bit
            
            # Propagation
            # 1. Remove val from domains of other cells in row i and col j
            affected_cells = []
            for j2 in range(n):
                if j2 == j: continue
                if val in domains[i][j2]:
                    domains[i][j2].remove(val)
                    affected_cells.append((i, j2))
            for i2 in range(n):
                if i2 == i: continue
                if val in domains[i2][j]:
                    domains[i2][j].remove(val)
                    affected_cells.append((i2, j))
                    
            # 2. Orthogonality Propagation
            ortho_affected = []
            for k, sq_k in enumerate(existing_squares):
                val_k = sq_k[i][j]
                pair = (val, val_k)
                if pair in pair_used[k]:
                    # Conflict! (Should be caught by domain pruning if we did it right, 
                    # but we didn't check pair_used in the loop above for the current cell's val 
                    # against existing pairs? 
                    # Actually, the domain pruning step 2 below handles future cells.
                    # But does the current cell's val conflict with an *existing* pair?
                    # The pair (val, val_k) must be unique.
                    # If it's already in pair_used[k], it means some other cell (i',j') already used this pair.
                    # This means we should have removed val from domains[i][j] when that other cell was assigned.
                    # So this check should never trigger if propagation is correct.
                    # But let's be safe.
                    pass
                    
                pair_used[k].add(pair)
                
                # For all cells (i2, j2) where sq_k[i2][j2] == val_k, remove val from domain
                for (i2, j2) in pos_in_sq[k][val_k]:
                    if i2 == i and j2 == j:
                        continue
                    if val in domains[i2][j2]:
                        domains[i2][j2].remove(val)
                        ortho_affected.append((i2, j2))
                        
            # Check for empty domains in affected cells
            valid = True
            for (i2, j2) in affected_cells + ortho_affected:
                if len(domains[i2][j2]) == 0:
                    valid = False
                    break
                    
            if valid:
                if backtrack():
                    return True
            else:
                # Undo
                pass # Undo logic below
                
            # Undo
            grid[i][j] = -1
            row_masks[i] &= ~bit
            col_masks[j] &= ~bit
            
            # Restore domains
            for (i2, j2) in affected_cells:
                domains[i2][j2].add(val)
            for (i2, j2) in ortho_affected:
                domains[i2][j2].add(val)
                
            for k in range(len(existing_squares)):
                val_k = existing_squares[k][i][j]
                pair_used[k].remove((val, val_k))
                
        return False

    if backtrack():
        return grid
    return None
