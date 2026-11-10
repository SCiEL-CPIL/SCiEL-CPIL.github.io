def _find_orthogonal(existing_squares, n, deadline):
    num_existing = len(existing_squares)
    diag_cells = [] # Not strictly needed if we access existing_squares directly, but might be faster?
    # Accessing existing_squares[i][r][c] is O(1).
    
    L = [[-1]*n for _ in range(n)]
    row_masks = [0]*n
    col_masks = [0]*n
    diag_masks = [[0]*n for _ in range(num_existing)]
    
    num_empty = n*n
    
    def backtrack():
        nonlocal num_empty
        if time.time() > deadline:
            return False
            
        if num_empty == 0:
            return True
            
        min_count = n + 1
        best_r = -1
        best_c = -1
        best_avail_mask = 0
        
        for r in range(n):
            for c in range(n):
                if L[r][c] == -1:
                    mask = row_masks[r] | col_masks[c]
                    for i in range(num_existing):
                        sym_in_existing = existing_squares[i][r][c]
                        mask |= diag_masks[i][sym_in_existing]
                        
                    avail = (~mask) & ((1 << n) - 1)
                    count = bin(avail).count('1')
                    if count < min_count:
                        min_count = count
                        best_r = r
                        best_c = c
                        best_avail_mask = avail
                        if count == 1:
                            break
            if min_count == 1:
                break
                
        if best_r == -1:
            return False
            
        if min_count == 0:
            return False
            
        r = best_r
        c = best_c
        avail = best_avail_mask
        
        syms = []
        temp = avail
        while temp:
            lowbit = temp & (-temp)
            sym = lowbit.bit_length() - 1
            syms.append(sym)
            temp ^= lowbit
            
        for sym in syms:
            bit = 1 << sym
            L[r][c] = sym
            row_masks[r] |= bit
            col_masks[c] |= bit
            for i in range(num_existing):
                sym_in_existing = existing_squares[i][r][c]
                diag_masks[i][sym_in_existing] |= bit
                
            num_empty -= 1
            if backtrack():
                return True
            num_empty += 1
            
            L[r][c] = -1
            row_masks[r] &= ~bit
            col_masks[c] &= ~bit
            for i in range(num_existing):
                sym_in_existing = existing_squares[i][r][c]
                diag_masks[i][sym_in_existing] &= ~bit
                
        return False

    if backtrack():
        return L
    return None
