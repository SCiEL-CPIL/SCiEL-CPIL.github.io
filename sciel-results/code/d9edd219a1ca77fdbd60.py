def construct():
    N = 18
    # Quadratic residues mod 17
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Construct Parity Check Matrix H (8x17)
    # Rows are cyclic shifts of v_res
    # v_res[i] = 1 if i in residues else 0
    # Note: Cyclic shift of length 17.
    
    # We will use integers to represent rows of H (17 bits)
    # Bit i corresponds to column i
    
    v_res = 0
    for i in residues:
        v_res |= (1 << i)
        
    H_rows = []
    v = v_res
    for _ in range(8):
        H_rows.append(v)
        # Cyclic shift right by 1 (mod 2^17 - 1 is not simple, just rotate)
        # Rotate right: (v >> 1) | ((v & 1) << 16)
        v = (v >> 1) | ((v & 1) << 16)
        
    # Gaussian Elimination to find Null Space Basis
    # We want to solve H * x = 0
    # H is 8x17.
    # We'll use the rows to eliminate leading bits.
    
    # Copy H_rows to a list for modification
    mat = H_rows[:]
    rank = 0
    for col in range(17):
        # Find pivot
        pivot_row = -1
        for r in range(rank, 8):
            if (mat[r] >> col) & 1:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
            
        # Swap pivot row with current rank row
        mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
        
        # Eliminate this column in other rows
        for r in range(8):
            if r != rank and ((mat[r] >> col) & 1):
                mat[r] ^= mat[rank]
                
        rank += 1
        if rank == 8:
            break
            
    # The basis for the null space can be found by setting free variables to 1 and pivot variables to 0 (and solving)
    # Or simpler: The rows of the matrix after elimination are linearly independent.
    # The null space dimension is 17 - rank = 17 - 8 = 9.
    
    # Let's identify pivot columns and free columns.
    pivot_cols = []
    free_cols = []
    for col in range(17):
        # Check if this column has a pivot in the row-reduced form
        # In our elimination, row i has pivot at some column.
        # We didn't store the pivot column explicitly in a clean way during the loop above because we iterated col.
        # Let's re-do or track.
        
        pass

    # Easier approach:
    # The generator matrix G can be derived.
    # Actually, we can just generate the codewords by finding the null space directly using a standard library or manual backtracking?
    # 17 variables, 8 equations.
    # We can iterate over the 9 free variables (2^9 = 512 combinations).
    # Which are the free variables?
    # In the reduced row echelon form (RREF), the columns that are NOT unit vectors are free.
    
    # Let's compute RREF properly.
    mat = H_rows[:]
    pivots = {} # col -> row
    for col in range(17):
        pivot_row = -1
        for r in range(len(mat)):
            if (mat[r] >> col) & 1:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        # Swap
        if pivot_row != len(pivots):
             mat[len(pivots)], mat[pivot_row] = mat[pivot_row], mat[len(pivots)]
        # Eliminate
        for r in range(len(mat)):
            if r != len(pivots) and ((mat[r] >> col) & 1):
                mat[r] ^= mat[len(pivots)]
        pivots[col] = len(pivots)
        len_pivots = len(pivots)
        
    # Now mat is in RREF.
    # Pivot columns are keys of pivots.
    # Free columns are the rest.
    free_cols = [c for c in range(17) if c not in pivots]
    
    # For each combination of free variables, compute the pivot variables.
    # x_pivot = sum(x_free * mat[row][col_free])
    
    words = []
    for mask in range(1 << len(free_cols)):
        c = 0
        # Set free variables
        for i, col in enumerate(free_cols):
            if (mask >> i) & 1:
                c |= (1 << col)
        
        # Compute pivot variables
        for p_col, p_row in pivots.items():
            # x_p_col = sum_{f in free} (mat[p_row] has 1 at f) * x_f
            val = 0
            row_bits = mat[p_row]
            # Iterate over free columns
            for i, f_col in enumerate(free_cols):
                if (row_bits >> f_col) & 1:
                    val ^= (mask >> i) & 1
            if val:
                c |= (1 << p_col)
                
        # c is the 17-bit codeword
        # Calculate parity
        parity = c.bit_count() % 2
        # Extend to 18 bits
        # Bit 17 is parity
        word_18 = c | (parity << 17)
        words.append(word_18)
        
    return words
