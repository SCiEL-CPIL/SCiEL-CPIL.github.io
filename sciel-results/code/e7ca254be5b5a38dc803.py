"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # We decompose 2x3x8 into:
    # Block L: A[0:2, 0:1] * B[0:1, 0:8] -> C[0:2, 0:8] (size 2x1x8)
    # Block R: A[0:2, 1:3] * B[1:3, 0:8] -> C[0:2, 0:8] (size 2x2x8)
    
    rank_L = 2 * 1 * 8  # 16
    rank_R = 28  # Strassen rank for 2x2x8
    
    rank = rank_L + rank_R  # 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # --- Block L (Schoolbook for 2x1x8) ---
    # i in 0..1, j=0, k in 0..7
    col = 0
    for i in range(2):
        for k in range(8):
            # U index: i*m + j = i*3 + 0 = 3i
            # V index: j*p + k = 0*8 + k = k
            # W index: k*n + i = k*2 + i
            U[3*i][col] = 1
            V[k][col] = 1
            W[2*k + i][col] = 1
            col += 1
            
    # --- Block R (Strassen for 2x2x8) ---
    # A_R is 2x2 (indices i in 0..1, j in 1..2 relative to global j)
    # B_R is 2x8 (indices j in 1..2, k in 0..7)
    # C_R is 2x8 (indices i in 0..1, k in 0..7)
    
    # Global indices:
    # A_R[i][j_rel] -> U[i*3 + (1+j_rel)]
    # B_R[j_rel][k] -> V[(1+j_rel)*8 + k]
    # C_R[i][k] -> W[k*2 + i]
    
    # Strassen factors M1..M7 for each k (since B is 2x8, we treat each column k independently?
    # No, Strassen applies to the matrix multiplication. Since B is 2x8, we can view it as 8 independent
    # multiplications of a 2x2 matrix A_R by a 2x1 vector B_R[:,k]? 
    # Actually, Strassen is for matrix multiplication. Here A_R is 2x2 and B_R is 2x8.
    # The product is A_R * B_R.
    # We can apply Strassen to the 2x2 * 2x2 blocks? 
    # Since B_R is 2x8, it can be split into four 2x2 blocks: B11, B12, B21, B22 where 
    # B11 is rows 1, cols 0,4? No, standard partitioning.
    # Let's partition B_R (2x8) into columns.
    # It's easier to think of A_R * B_R as 8 independent computations of A_R * b_k where b_k is 2x1.
    # But Strassen reduces the number of scalar multiplications for 2x2 * 2x2.
    # If we view B_R as 4 blocks of 2x2: B[0:2, 0:2], B[0:2, 2:4], etc.
    # Then A_R (2x2) * B_block (2x2) can be done with 7 multiplications.
    # There are 4 such blocks (8 cols / 2).
    # So total rank for this part is 4 * 7 = 28.
    
    # Let's verify indices for one 2x2 block of B_R.
    # B_R has rows j_rel=0,1 (global j=1,2) and cols k=0..7.
    # Block 0: k=0,1. B[0,0], B[0,1], B[1,0], B[1,1].
    # Block 1: k=2,3.
    # Block 2: k=4,5.
    # Block 3: k=6,7.
    
    for b_idx in range(4):
        k_start = b_idx * 2
        k1 = k_start
        k2 = k_start + 1
        
        # Indices in B_R for this 2x2 block:
        # Row 0 (j_rel=0): cols k1, k2
        # Row 1 (j_rel=1): cols k1, k2
        
        # Strassen Products M1..M7:
        # M1 = A00 * (B00 + B11)
        # M2 = (A10 + A00) * B01
        # M3 = A00 * (B10 - B00)
        # M4 = A11 * (B01 - B11)
        # M5 = (A00 + A11) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Note: My previous notes used different notation. Let's stick to standard:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        
        # In our case:
        # A00 = A[0,1] (global i=0, j=1) -> U index 0*3+1 = 1
        # A01 = A[0,2] (global i=0, j=2) -> U index 0*3+2 = 2
        # A10 = A[1,1] (global i=1, j=1) -> U index 1*3+1 = 4
        # A11 = A[1,2] (global i=1, j=2) -> U index 1*3+2 = 5
        
        # B00 = B[1, k1] (global j=1, k=k1) -> V index 1*8+k1 = 8+k1
        # B01 = B[1, k2] (global j=1, k=k2) -> V index 1*8+k2 = 8+k2
        # B10 = B[2, k1] (global j=2, k=k1) -> V index 2*8+k1 = 16+k1
        # B11 = B[2, k2] (global j=2, k=k2) -> V index 2*8+k2 = 16+k2
        
        # Output C:
        # C00 (local) = M1 + M4 - M5 + M7 -> Global C[0, k1] -> W index k1*2+0 = 2*k1
        # C01 (local) = M3 + M5 -> Global C[0, k2] -> W index k2*2+0 = 2*k2
        # C10 (local) = M2 + M4 -> Global C[1, k1] -> W index k1*2+1 = 2*k1+1
        # C11 (local) = M1 - M2 + M3 + M6 -> Global C[1, k2] -> W index k2*2+1 = 2*k2+1
        
        # We will add 7 columns for this block.
        for s in range(7):
            col_idx = col + s
            
            # Determine u, v, w vectors for this Strassen term
            
            # u vector (length 6), v vector (length 24), w vector (length 16)
            # We only need to set the non-zero elements.
            
            if s == 0: # M1 = A00 * (B00 + B11)
                # U: A00 -> idx 1
                U[1][col_idx] = 1
                # V: B00 -> idx 8+k1, B11 -> idx 16+k2
                V[8+k1][col_idx] = 1
                V[16+k2][col_idx] = 1
                # W: C00 -> idx 2*k1, C11 -> idx 2*k2+1
                W[2*k1][col_idx] = 1
                W[2*k2+1][col_idx] = 1
            elif s == 1: # M2 = (A10 + A00) * B01
                # U: A10 -> idx 4, A00 -> idx 1
                U[4][col_idx] = 1
                U[1][col_idx] = 1
                # V: B01 -> idx 8+k2
                V[8+k2][col_idx] = 1
                # W: C10 -> idx 2*k1+1, C11 -> idx 2*k2+1 (-1)
                W[2*k1+1][col_idx] = 1
                W[2*k2+1][col_idx] = -1
            elif s == 2: # M3 = A00 * (B10 - B00)
                # U: A00 -> idx 1
                U[1][col_idx] = 1
                # V: B10 -> idx 16+k1, B00 -> idx 8+k1 (-1)
                V[16+k1][col_idx] = 1
                V[8+k1][col_idx] = -1
                # W: C01 -> idx 2*k2, C11 -> idx 2*k2+1
                W[2*k2][col_idx] = 1
                W[2*k2+1][col_idx] = 1
            elif s == 3: # M4 = A11 * (B01 - B11)
                # U: A11 -> idx 5
                U[5][col_idx] = 1
                # V: B01 -> idx 8+k2, B11 -> idx 16+k2 (-1)
                V[8+k2][col_idx] = 1
                V[16+k2][col_idx] = -1
                # W: C00 -> idx 2*k1, C10 -> idx 2*k1+1
                W[2*k1][col_idx] = 1
                W[2*k1+1][col_idx] = 1
            elif s == 4: # M5 = (A00 + A11) * B11
                # U: A00 -> idx 1, A11 -> idx 5
                U[1][col_idx] = 1
                U[5][col_idx] = 1
                # V: B11 -> idx 16+k2
                V[16+k2][col_idx] = 1
                # W: C00 -> idx 2*k1 (-1), C01 -> idx 2*k2
                W[2*k1][col_idx] = -1
                W[2*k2][col_idx] = 1
            elif s == 5: # M6 = (A10 - A00) * (B00 + B01)
                # U: A10 -> idx 4, A00 -> idx 1 (-1)
                U[4][col_idx] = 1
                U[1][col_idx] = -1
                # V: B00 -> idx 8+k1, B01 -> idx 8+k2
                V[8+k1][col_idx] = 1
                V[8+k2][col_idx] = 1
                # W: C11 -> idx 2*k2+1
                W[2*k2+1][col_idx] = 1
            elif s == 6: # M7 = (A01 - A11) * (B10 + B11)
                # U: A01 -> idx 2, A11 -> idx 5 (-1)
                U[2][col_idx] = 1
                U[5][col_idx] = -1
                # V: B10 -> idx 16+k1, B11 -> idx 16+k2
                V[16+k1][col_idx] = 1
                V[16+k2][col_idx] = 1
                # W: C00 -> idx 2*k1
                W[2*k1][col_idx] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

