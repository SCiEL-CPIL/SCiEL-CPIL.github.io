"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # But actually the tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And we need this to equal 1 for the matrix multiplication tensor.
    
    # Actually, the correct check is:
    # C = A * B where C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (all ones) for the standard matrix multiplication.
    # So sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False, (i, j, k, s)
    return True, None

def construct():
    """Construct a decomposition for 3x3x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    start_time = time.time()
    deadline = start_time + 120.0  # Must finish within 120s
    
    # Strategy: Try to find rank-45 decomposition using three rank-15 blocks for 3x3x2.
    # If that fails or takes too long, fall back to schoolbook rank 54.
    
    # First, try rank-45 construction
    try:
        # Construct rank-15 decomposition for 3x3x2
        # Using a systematic approach with verification
        
        rank_block = 15
        U_block = [[0]*rank_block for _ in range(9)]
        V_block = [[0]*rank_block for _ in range(6)]
        W_block = [[0]*rank_block for _ in range(6)]
        
        # Use Strassen for 2x2x2 subproblem (rows 0,1 of A; cols 0,1 of B)
        # Then handle remaining terms
        
        # Strassen's 7 terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # M1 = (A00+A11)(B00+B11)
        U_block[0][0] = 1; U_block[4][0] = 1
        V_block[0][0] = 1; V_block[3][0] = 1
        W_block[0][0] = 1; W_block[3][0] = 1; W_block[1][0] = 1; W_block[4][0] = 1
        
        # M2 = (A10+A11)B00
        U_block[3][1] = 1; U_block[4][1] = 1
        V_block[0][1] = 1
        W_block[1][1] = 1; W_block[4][1] = 1
        
        # M3 = A00(B10-B11)
        U_block[0][2] = 1
        V_block[2][2] = 1; V_block[3][2] = -1
        W_block[3][2] = 1
        
        # M4 = A11(B01-B00)
        U_block[4][3] = 1
        V_block[1][3] = 1; V_block[0][3] = -1
        W_block[0][3] = 1; W_block[1][3] = 1
        
        # M5 = (A00+A10)B11
        U_block[0][4] = 1; U_block[3][4] = 1
        V_block[3][4] = 1
        W_block[0][4] = -1; W_block[1][4] = 1
        
        # M6 = (A01-A00)(B00+B10)
        U_block[1][5] = 1; U_block[0][5] = -1
        V_block[0][5] = 1; V_block[2][5] = 1
        W_block[4][5] = 1
        
        # M7 = (A10-A11)(B01+B11)
        U_block[3][6] = 1; U_block[4][6] = -1
        V_block[1][6] = 1; V_block[3][6] = 1
        W_block[0][6] = 1; W_block[1][6] = -1
        
        # Now handle remaining terms (involving row 2 of A or col 2 of B)
        # C[0,0] += A[0,2]*B[2,0]
        U_block[2][7] = 1; V_block[4][7] = 1; W_block[0][7] = 1
        
        # C[0,1] += A[0,2]*B[2,1]
        U_block[2][8] = 1; V_block[5][8] = 1; W_block[3][8] = 1
        
        # C[1,0] += A[1,2]*B[2,0]
        U_block[5][9] = 1; V_block[4][9] = 1; W_block[1][9] = 1
        
        # C[1,1] += A[1,2]*B[2,1]
        U_block[5][10] = 1; V_block[5][10] = 1; W_block[4][10] = 1
        
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
        
        # Use 3 terms for row 2:
        # T1: A[2,0] * B[0,0] -> C[2,0]
        U_block[6][11] = 1; V_block[0][11] = 1; W_block[2][11] = 1
        
        # T2: A[2,0] * B[0,1] -> C[2,1]
        U_block[6][12] = 1; V_block[1][12] = 1; W_block[5][12] = 1
        
        # T3: A[2,1] * B[1,0] -> C[2,0]
        U_block[7][13] = 1; V_block[2][13] = 1; W_block[2][13] = 1
        
        # T4: A[2,1] * B[1,1] -> C[2,1]
        U_block[7][14] = 1; V_block[3][14] = 1; W_block[5][14] = 1
        
        # Missing: A[2,2]*B[2,0] -> C[2,0] and A[2,2]*B[2,1] -> C[2,1]
        # We're out of terms! Need to combine.
        
        # Let me try a different approach: use a known valid rank-15 construction.
        # After research, here's a valid rank-15 construction for 3x3x2:
        
        # Clear and rebuild
        U_block = [[0]*15 for _ in range(9)]
        V_block = [[0]*15 for _ in range(6)]
        W_block = [[0]*15 for _ in range(6)]
        
        # Use a different decomposition strategy:
        # For each output row i, compute C[i,:] = A[i,:] * B
        # This is a 1x3x2 problem for each i, which has rank 3.
        # Total: 3*3 = 9, but we can share some computations.
        
        # Actually, let me just use schoolbook for the block (rank 18) and combine 3 blocks = rank 54.
        # But we want rank 45. Let me try rank 15 per block.
        
        # After careful analysis, here's a valid rank-15 construction:
        
        # Terms 0-6: Strassen for 2x2x2 (A[0:2,0:2], B[0:2,0:2] -> C[0:2,0:2])
        # Already set above, let me keep them.
        
        # For the remaining 8 terms, I need to handle:
        # - A[0,2]*B[2,0] -> C[0,0]
        # - A[0,2]*B[2,1] -> C[0,1]  
        # - A[1,2]*B[2,0] -> C[1,0]
        # - A[1,2]*B[2,1] -> C[1,1]
        # - A[2,0]*B[0,0] -> C[2,0]
        # - A[2,0]*B[0,1] -> C[2,1]
        # - A[2,1]*B[1,0] -> C[2,0]
        # - A[2,1]*B[1,1] -> C[2,1]
        # - A[2,2]*B[2,0] -> C[2,0]
        # - A[2,2]*B[2,1] -> C[2,1]
        
        # That's 10 monomials. I need to represent them in 8 rank-1 terms.
        # I can combine pairs that share the same A entry or B entry.
        
        # Combine: A[2,2]*B[2,0] and A[2,2]*B[2,1] can't be combined into one rank-1 term
        # because they go to different output positions.
        
        # Let me try: use 2 terms for (A[2,2]*B[2,0], A[2,2]*B[2,1])
        # and 2 terms for (A[2,0]*B[0,0], A[2,0]*B[0,1])
        # and 2 terms for (A[2,1]*B[1,0], A[2,1]*B[1,1])
        # and 2 terms for (A[0,2]*B[2,0], A[1,2]*B[2,0])
        # and 2 terms for (A[0,2]*B[2,1], A[1,2]*B[2,1])
        # That's 10 terms, but I only have 8 slots.
        
        # I think rank 15 is not achievable with this simple approach.
        # Let me fall back to schoolbook rank 54.
        
        rank_total = n * m * p  # 54
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Verify
        valid, _ = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {"U": U, "V": V, "W": W}
        
    except Exception as e:
        pass
    
    # Fallback: schoolbook
    rank_total = n * m * p  # 54
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
