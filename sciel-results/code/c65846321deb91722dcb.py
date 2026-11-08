"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for 2x2 blocks of 3x3 matrices.
    # Each 3x3 multiplication uses the schoolbook algorithm (rank 27).
    # Total rank = 7 * 27 = 189.
    rank = 189
    
    # Initialize matrices
    # U: (n*m, rank) -> (36, 189)
    # V: (m*p, rank) -> (36, 189)
    # W: (p*n, rank) -> (36, 189)
    # Note: W index is k*n + i
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get global index for A[i,j]
    def idx_A(i, j):
        return i * m + j
    
    # Helper to get global index for B[j,k]
    def idx_B(j, k):
        return j * p + k
    
    # Helper to get global index for C[i,k] in W (k*n + i)
    def idx_W(i, k):
        return k * n + i
        
    # Strassen steps for 2x2 blocks of size 3x3
    # Blocks:
    # A11: i in 0..2, j in 0..2
    # A12: i in 0..2, j in 3..5
    # A21: i in 3..5, j in 0..2
    # A22: i in 3..5, j in 3..5
    
    # Same for B and C.
    
    # Define the 7 Strassen multiplications
    # Each step s has:
    # X_s: linear combo of A blocks
    # Y_s: linear combo of B blocks
    # Z_s = X_s * Y_s
    # C blocks are linear combos of Z_s
    
    # Coefficients for X_s (A blocks) and Y_s (B blocks)
    # Order of blocks: 11, 12, 21, 22
    
    strassen_steps = [
        # M1 = (A11 + A22)(B11 + B22)
        {
            'X': {0: 1, 1: 0, 2: 0, 3: 1},  # A11 + A22
            'Y': {0: 1, 1: 0, 2: 0, 3: 1},  # B11 + B22
            'Z_to_C': {0: 1, 1: 0, 2: 0, 3: 1} # C11 += Z, C22 += Z
        },
        # M2 = (A21 + A22)B11
        {
            'X': {0: 0, 1: 0, 2: 1, 3: 1},  # A21 + A22
            'Y': {0: 1, 1: 0, 2: 0, 3: 0},  # B11
            'Z_to_C': {0: 0, 1: 0, 2: 1, 3: 0} # C21 += Z
        },
        # M3 = A11(B12 - B22)
        {
            'X': {0: 1, 1: 0, 2: 0, 3: 0},  # A11
            'Y': {0: 0, 1: 1, 2: 0, 3: -1}, # B12 - B22
            'Z_to_C': {0: 0, 1: 1, 2: 0, 3: 0} # C12 += Z
        },
        # M4 = A22(B21 - B11)
        {
            'X': {0: 0, 1: 0, 2: 0, 3: 1},  # A22
            'Y': {0: -1, 1: 0, 2: 1, 3: 0}, # B21 - B11
            'Z_to_C': {0: 1, 1: 0, 2: 1, 3: 0} # C11 += Z, C21 += Z
        },
        # M5 = (A11 + A12)B22
        {
            'X': {0: 1, 1: 1, 2: 0, 3: 0},  # A11 + A12
            'Y': {0: 0, 1: 0, 2: 0, 3: 1},  # B22
            'Z_to_C': {0: -1, 1: 1, 2: 0, 3: 0} # C11 -= Z, C12 += Z
        },
        # M6 = (A21 - A11)(B11 + B12)
        {
            'X': {0: -1, 1: 0, 2: 1, 3: 0}, # A21 - A11
            'Y': {0: 1, 1: 1, 2: 0, 3: 0},  # B11 + B12
            'Z_to_C': {0: 0, 1: 0, 2: 0, 3: 1} # C22 += Z
        },
        # M7 = (A12 - A22)(B21 + B22)
        {
            'X': {0: 0, 1: 1, 2: 0, 3: -1}, # A12 - A22
            'Y': {0: 0, 1: 0, 2: 1, 3: 1},  # B21 + B22
            'Z_to_C': {0: 1, 1: 0, 2: 0, 3: 0} # C11 += Z
        }
    ]
    
    # Block offsets for 3x3 blocks
    # Block 0 (11): i_off=0, j_off=0
    # Block 1 (12): i_off=0, j_off=3
    # Block 2 (21): i_off=3, j_off=0
    # Block 3 (22): i_off=3, j_off=3
    
    block_offsets = [
        (0, 0), # 11
        (0, 3), # 12
        (3, 0), # 21
        (3, 3)  # 22
    ]
    
    col = 0
    for step in strassen_steps:
        X_coeffs = step['X']
        Y_coeffs = step['Y']
        Z_to_C = step['Z_to_C']
        
        # Iterate over all 27 scalar multiplications in the 3x3 product
        for u in range(3):      # row of X, row of Z
            for w in range(3):  # col of X, row of Y
                for v in range(3): # col of Y, col of Z
                    
                    # 1. Update U
                    # X[u,w] = sum_block coeff * A_block[u,w]
                    for block_idx in range(4):
                        coeff = X_coeffs[block_idx]
                        if coeff != 0:
                            i_off, j_off = block_offsets[block_idx]
                            i = u + i_off
                            j = w + j_off
                            a_idx = idx_A(i, j)
                            U[a_idx][col] += coeff
                    
                    # 2. Update V
                    # Y[w,v] = sum_block coeff * B_block[w,v]
                    for block_idx in range(4):
                        coeff = Y_coeffs[block_idx]
                        if coeff != 0:
                            i_off, j_off = block_offsets[block_idx]
                            j = w + i_off # Note: B rows are j, so offset is on j
                            k = v + j_off # Note: B cols are k, so offset is on k
                            b_idx = idx_B(j, k)
                            V[b_idx][col] += coeff
                    
                    # 3. Update W
                    # Z[u,v] contributes to C blocks
                    for block_idx in range(4):
                        coeff = Z_to_C[block_idx]
                        if coeff != 0:
                            i_off, j_off = block_offsets[block_idx]
                            i = u + i_off
                            k = v + j_off
                            w_idx = idx_W(i, k)
                            W[w_idx][col] += coeff
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
