"""Winograd-style decomposition for 3x4 * 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    We check that for all i in [0,n), k in [0,p):
    sum_{j in [0,m)} sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    And for all other combinations, the sum is 0.
    
    Actually, the tensor T is defined such that:
    T_{a, b, c} = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
    Otherwise 0.
    
    The decomposition is sum_r U[a,r] V[b,r] W[c,r] = T_{a,b,c}.
    """
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Construct the target tensor implicitly by checking the condition
    # We iterate over all possible (a, b, c) combinations? That's too slow (12*16*12 = 2304)
    # Instead, we can check the reconstruction of the multiplication.
    
    # Let's check: C[k,i] = sum_j A[i,j] B[j,k]
    # With decomposition: C[k,i] = sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r]
    
    # Let's pick random A and B and check if C = A @ B (transposed appropriately)
    # A is 3x4, B is 4x4. C should be 4x3 (based on W index k*n+i).
    # C[k,i] = sum_j A[i,j] B[j,k]
    
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    
    # Compute standard C
    C_std = np.zeros((p, n))
    for i in range(n):
        for k in range(p):
            C_std[k, i] = np.dot(A[i, :], B[:, k])
    
    # Compute decomposed C
    C_dec = np.zeros((p, n))
    for i in range(n):
        for k in range(p):
            # C_dec[k,i] = sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r]
            # We can compute the intermediate vector for each r?
            # Or just sum over j and r.
            
            # Efficient way:
            # M[i, k] = sum_j U[i*m+j, :] * V[j*p+k, :]  (element-wise product then sum over j)
            # Then C_dec[k, i] = sum_r M[i, k, r] * W[k*n+i, r]
            
            # Let's do it directly for clarity in verification
            val = 0.0
            for j in range(m):
                u_vec = U_arr[i*m+j, :]
                v_vec = V_arr[j*p+k, :]
                # Product u_vec * v_vec is a vector of length rank
                # Sum over r of (u_vec * v_vec) * W[k*n+i, r]
                # = dot(u_vec * v_vec, W[k*n+i, :])
                val += np.dot(u_vec * v_vec, W_arr[k*n+i, :])
            C_dec[k, i] = val
            
    if not np.allclose(C_std, C_dec, atol=tol):
        return False
    
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initialize with schoolbook (rank 48)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Strategy: Try to find a lower rank decomposition using random search + refinement?
    # Given the small dimensions, we can try to construct a specific algebraic decomposition.
    # 
    # Let's try the Rank 44 approach from Inspiration 3 more concretely if possible,
    # or a simpler Rank 40/42 approach.
    #
    # Actually, let's try to implement a specific Winograd-based reduction.
    # 
    # Winograd for 2x2: Rank 7 instead of 8.
    # We have 3x4 * 4x4.
    # 
    # If we can group the 4 columns into two groups of 2, and use Winograd on the 2x2 blocks...
    # 
    # Let's attempt a construction for Rank 44 using Strassen for 2x2 blocks.
    #
    # However, coding a verified Strassen decomposition from scratch is error-prone.
    # Let's try a different tactic: Use the known Strassen 2x2 factors and embed them.
    #
    # Strassen 2x2 factors:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # We can use this to multiply 2x2 matrices.
    #
    # Our problem: 3x4 * 4x4.
    # Let's split A into A_top (2x4) and A_bot (1x4).
    # Split B into 4x2 blocks? Or 2x2 blocks.
    #
    # Let's try to use Strassen to compute A_top * B (2x4 * 4x4 -> 2x4).
    # Split A_top into [A_TL, A_TR] (2x2 each).
    # Split B into [[B_TL, B_TR], [B_BL, B_BR]] (2x2 each).
    #
    # C_TL = A_TL * B_TL + A_TR * B_BL
    # C_TR = A_TL * B_TR + A_TR * B_BR
    #
    # This involves 4 matrix multiplications of 2x2 * 2x2.
    # Each 2x2*2x2 can be done in rank 7 (Strassen).
    # Total rank = 4 * 7 = 28 for the top part.
    # Bottom part (1x4 * 4x4 -> 1x4) is rank 16 (schoolbook).
    # Total rank = 28 + 16 = 44.
    #
    # Let's construct the factors for this Rank 44 decomposition.
    
    # We will build U, V, W for the 44 rank case.
    #
    # Indices:
    # U: (12, 44)
    # V: (16, 44)
    # W: (12, 44)
    #
    # Columns 0-27: Top part (Strassen for 2x4*4x4)
    # Columns 28-43: Bottom part (Schoolbook for 1x4*4x4)
    
    U_44 = [[0] * 44 for _ in range(12)]
    V_44 = [[0] * 44 for _ in range(16)]
    W_44 = [[0] * 44 for _ in range(12)]
    
    # --- Bottom Part (Rows 2 of A, i.e., i=2) ---
    # A_bot is row 2 of A. Index in A: 2*4 + j = 8 + j.
    # B is 4x4. Index in B: j*4 + k.
    # C_bot is row 2 of C (in the n dimension). Index in C: k*3 + 2.
    #
    # Schoolbook for 1x4*4x4:
    # For j in 0..3, k in 0..3:
    #   U[8+j, col] = 1
    #   V[j*4+k, col] = 1
    #   W[k*3+2, col] = 1
    
    col = 28
    for j in range(4):
        for k in range(4):
            U_44[8 + j][col] = 1
            V_44[j * 4 + k][col] = 1
            W_44[k * 3 + 2][col] = 1
            col += 1
    
    # --- Top Part (Rows 0,1 of A, i.e., i=0,1) ---
    # A_top is 2x4. Split into A_TL (2x2, cols 0,1) and A_TR (2x2, cols 2,3).
    # B is 4x4. Split into 2x2 blocks:
    # B_TL (rows 0,1; cols 0,1), B_TR (rows 0,1; cols 2,3)
    # B_BL (rows 2,3; cols 0,1), B_BR (rows 2,3; cols 2,3)
    #
    # C_TL (2x2, cols 0,1 of C) = A_TL * B_TL + A_TR * B_BL
    # C_TR (2x2, cols 2,3 of C) = A_TL * B_TR + A_TR * B_BR
    #
    # We need to compute 4 products of 2x2 matrices using Strassen.
    #
    # Let's define the Strassen factors for a generic 2x2 multiplication X*Y -> Z.
    # X is 2x2, Y is 2x2.
    # Indices in X: 0,1,2,3 (row-major: 00,01,10,11)
    # Indices in Y: 0,1,2,3
    # Indices in Z: 0,1,2,3
    #
    # Strassen M's:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # We have 4 such multiplications.
    # 1. A_TL * B_TL -> contributes to C_TL
    # 2. A_TR * B_BL -> contributes to C_TL
    # 3. A_TL * B_TR -> contributes to C_TR
    # 4. A_TR * B_BR -> contributes to C_TR
    
    # We will assign 7 columns to each multiplication.
    # Total 28 columns.
    
    def add_strassen_block(U, V, W, base_col, 
                           a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, 
                           n, m, p):
        """
        Adds Strassen factors for multiplying A_sub (a_rows x a_cols) by B_sub (b_rows x b_cols)
        into C_sub (c_rows x c_cols).
        Here all are 2x2.
        
        a_rows: list of 2 row indices in A (i values)
        a_cols: list of 2 col indices in A (j values)
        b_rows: list of 2 row indices in B (j values)
        b_cols: list of 2 col indices in B (k values)
        c_rows: list of 2 row indices in C (k values)
        c_cols: list of 2 col indices in C (i values)
        
        Note: C is indexed by k (row) and i (col) in the flattened W[k*n+i].
        """
        
        # Helper to get U index for A[i,j]
        def u_idx(i, j): return i * m + j
        # Helper to get V index for B[j,k]
        def v_idx(j, k): return j * p + k
        # Helper to get W index for C[k,i]
        def w_idx(k, i): return k * n + i
        
        # X indices in local 2x2:
        # X00 -> A[a_rows[0], a_cols[0]]
        # X01 -> A[a_rows[0], a_cols[1]]
        # X10 -> A[a_rows[1], a_cols[0]]
        # X11 -> A[a_rows[1], a_cols[1]]
        
        x00 = u_idx(a_rows[0], a_cols[0])
        x01 = u_idx(a_rows[0], a_cols[1])
        x10 = u_idx(a_rows[1], a_cols[0])
        x11 = u_idx(a_rows[1], a_cols[1])
        
        # Y indices in local 2x2:
        # Y00 -> B[b_rows[0], b_cols[0]]
        # Y01 -> B[b_rows[0], b_cols[1]]
        # Y10 -> B[b_rows[1], b_cols[0]]
        # Y11 -> B[b_rows[1], b_cols[1]]
        
        y00 = v_idx(b_rows[0], b_cols[0])
        y01 = v_idx(b_rows[0], b_cols[1])
        y10 = v_idx(b_rows[1], b_cols[0])
        y11 = v_idx(b_rows[1], b_cols[1])
        
        # Z indices in local 2x2:
        # Z00 -> C[c_rows[0], c_cols[0]]
        # Z01 -> C[c_rows[0], c_cols[1]]
        # Z10 -> C[c_rows[1], c_cols[0]]
        # Z11 -> C[c_rows[1], c_cols[1]]
        
        z00 = w_idx(c_rows[0], c_cols[0])
        z01 = w_idx(c_rows[0], c_cols[1])
        z10 = w_idx(c_rows[1], c_cols[0])
        z11 = w_idx(c_rows[1], c_cols[1])
        
        # M1: (X00+X11)(Y00+Y11)
        col = base_col
        U[x00][col] += 1; U[x11][col] += 1
        V[y00][col] += 1; V[y11][col] += 1
        W[z00][col] += 1; W[z11][col] += 1
        col += 1
        
        # M2: (X10+X11)Y00
        U[x10][col] += 1; U[x11][col] += 1
        V[y00][col] += 1
        W[z10][col] += 1; W[z11][col] -= 1
        col += 1
        
        # M3: X00(Y01-Y11)
        U[x00][col] += 1
        V[y01][col] += 1; V[y11][col] -= 1
        W[z01][col] += 1; W[z11][col] += 1
        col += 1
        
        # M4: X11(Y10-Y00)
        U[x11][col] += 1
        V[y10][col] += 1; V[y00][col] -= 1
        W[z00][col] += 1; W[z10][col] += 1
        col += 1
        
        # M5: (X00+X01)Y11
        U[x00][col] += 1; U[x01][col] += 1
        V[y11][col] += 1
        W[z00][col] -= 1; W[z01][col] += 1
        col += 1
        
        # M6: (X10-X00)(Y00+Y01)
        U[x10][col] += 1; U[x00][col] -= 1
        V[y00][col] += 1; V[y01][col] += 1
        W[z11][col] += 1
        col += 1
        
        # M7: (X01-X11)(Y10+Y11)
        U[x01][col] += 1; U[x11][col] -= 1
        V[y10][col] += 1; V[y11][col] += 1
        W[z00][col] += 1
        col += 1
        
        return col
    
    # Block 1: A_TL * B_TL
    # A_TL rows: 0,1. Cols: 0,1.
    # B_TL rows: 0,1. Cols: 0,1.
    # C_TL rows (k): 0,1. Cols (i): 0,1.
    # Note: C_TL corresponds to C[k,i] for k in 0,1 and i in 0,1.
    
    # Wait, C is 4x3 (p x n).
    # C[k,i] where k is 0..3 (p), i is 0..2 (n).
    # A_TL is rows 0,1 of A (i=0,1), cols 0,1 (j=0,1).
    # B_TL is rows 0,1 of B (j=0,1), cols 0,1 (k=0,1).
    # Result C_TL should be for k in 0,1 and i in 0,1.
    
    # a_rows = [0, 1] (i indices)
    # a_cols = [0, 1] (j indices)
    # b_rows = [0, 1] (j indices)
    # b_cols = [0, 1] (k indices)
    # c_rows = [0, 1] (k indices)
    # c_cols = [0, 1] (i indices)
    
    col = add_strassen_block(U_44, V_44, W_44, 0, 
                             [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 
                             n, m, p)
    
    # Block 2: A_TR * B_BL
    # A_TR rows: 0,1. Cols: 2,3.
    # B_BL rows: 2,3. Cols: 0,1.
    # Result C_TL (same as above) k in 0,1, i in 0,1.
    
    col = add_strassen_block(U_44, V_44, W_44, col,
                             [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1],
                             n, m, p)
    
    # Block 3: A_TL * B_TR
    # A_TL rows: 0,1. Cols: 0,1.
    # B_TR rows: 0,1. Cols: 2,3.
    # Result C_TR k in 2,3, i in 0,1.
    
    col = add_strassen_block(U_44, V_44, W_44, col,
                             [0, 1], [0, 1], [0, 1], [2, 3], [2, 3], [0, 1],
                             n, m, p)
    
    # Block 4: A_TR * B_BR
    # A_TR rows: 0,1. Cols: 2,3.
    # B_BR rows: 2,3. Cols: 2,3.
    # Result C_TR k in 2,3, i in 0,1.
    
    col = add_strassen_block(U_44, V_44, W_44, col,
                             [0, 1], [2, 3], [2, 3], [2, 3], [2, 3], [0, 1],
                             n, m, p)
    
    # Verify Rank 44
    if _verify(U_44, V_44, W_44, n, m, p):
        best_U, best_V, best_W = U_44, V_44, W_44
        best_rank = 44
    else:
        # If verification fails, stick to schoolbook
        pass
    
    # If we have time, we could try to search for rank < 44, but 44 is a solid improvement over 48.
    # Given the complexity, we return the best verified result.
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
