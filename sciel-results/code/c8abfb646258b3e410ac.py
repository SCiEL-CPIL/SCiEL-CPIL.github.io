import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 10
    symbols = list(range(n))
    FULL_MASK = (1 << n) - 1
    
    # 1. Generate the first Latin Square (Cyclic)
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L1]
    
    # Check if we have time to search for a second square
    if time.time() > deadline - 5.0:
        return squares

    # 2. Search for a second Latin Square L2 orthogonal to L1
    # Using bitmask-based backtracking for speed
    
    L2 = [[None for _ in range(n)] for _ in range(n)]
    
    row_mask = [0] * n
    col_mask = [0] * n
    ortho_mask = [0] * n
    
    # Pre-compute L1 symbol at each cell
    # l1_val[r][c] = L1[r][c]
    
    def backtrack():
        if time.time() > deadline:
            return False
            
        # MRV: Find cell with min candidates
        best_r = -1
        best_c = -1
        min_cands = n + 1
        best_cand_mask = 0
        
        for r in range(n):
            for c in range(n):
                if L2[r][c] is None:
                    # Candidates are symbols NOT in row, col, or ortho mask
                    cand_mask = ~(row_mask[r] | col_mask[c] | ortho_mask[L1[r][c]]) & FULL_MASK
                    # Count bits
                    # Python 3.10+ has int.bit_count()
                    # Fallback for older: bin(x).count('1')
                    try:
                        cnt = cand_mask.bit_count()
                    except AttributeError:
                        cnt = bin(cand_mask).count('1')
                    
                    if cnt < min_cands:
                        min_cands = cnt
                        best_r = r
                        best_c = c
                        best_cand_mask = cand_mask
                        if min_cands == 1:
                            break
            if min_cands == 1:
                break
                
        if best_r == -1:
            return True
            
        if min_cands == 0:
            return False
            
        # Iterate through candidates in best_cand_mask
        mask = best_cand_mask
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            mask -= lsb
            v = lsb.bit_length() - 1
            
            # Assign
            L2[best_r][best_c] = v
            row_mask[best_r] |= lsb
            col_mask[best_c] |= lsb
            ortho_mask[L1[best_r][best_c]] |= lsb
            
            # Check consistency (forward checking)
            consistent = True
            # Check if any other empty cell has 0 candidates
            # Optimization: We can skip full forward checking if we trust MRV + backtracking
            # for n=10, it's usually fast enough without deep FC.
            # But let's do a quick check for empty domains.
            for r in range(n):
                if L2[best_r][best_c] is not None and r == best_r: continue # Skip assigned? No, check all unassigned.
            # Actually, simple backtracking with MRV is very fast for n=10.
            # Let's just recurse.
            
            if backtrack():
                return True
            
            # Undo
            L2[best_r][best_c] = None
            row_mask[best_r] &= ~lsb
            col_mask[best_c] &= ~lsb
            ortho_mask[L1[best_r][best_c]] &= ~lsb
            
        return False

    if backtrack():
        # Verify
        valid = True
        for r in range(n):
            if sorted(L2[r]) != symbols:
                valid = False
                break
        if valid:
            for c in range(n):
                col = [L2[r][c] for r in range(n)]
                if sorted(col) != symbols:
                    valid = False
                    break
        if valid:
            seen = set()
            for r in range(n):
                for c in range(n):
                    p = (L1[r][c], L2[r][c])
                    if p in seen:
                        valid = False
                        break
                    seen.add(p)
                if not valid:
                    break
            if valid:
                squares.append(L2)
                
    return squares
