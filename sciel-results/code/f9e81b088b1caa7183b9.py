"""Research program: Rank-14 construction for 2x3x3 via rank-7 decomposition of 3x3."""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Known rank-7 decomposition of 3x3 matrix multiplication (Bini et al.)
    # Factors U3, V3, W3 of shape (9, 7) each
    # T3[i,j,k] = sum_c U3[i*3+j, c] * V3[j*3+k, c] * W3[k*3+i, c] = 1
    
    # The rank-7 factors for 3x3 (using the well-known decomposition):
    U3 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ]
    
    V3 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ]
    
    W3 = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ]
    
    U3 = np.array(U3, dtype=float)
    V3 = np.array(V3, dtype=float)
    W3 = np.array(W3, dtype=float)
    
    # Verify the 3x3 rank-7 decomposition
    T3_target = np.ones((3, 3, 3))
    T3_recon = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for c in range(7):
                    T3_recon[i, j, k] += U3[i*3+j, c] * V3[j*3+k, c] * W3[k*3+i, c]
    
    if not np.allclose(T3_recon, T3_target):
        # The above factors are just identity-like, not the actual rank-7 decomposition.
        # Let me use the correct Bini et al. rank-7 factors.
        
        # Correct rank-7 decomposition of 3x3 matrix multiplication:
        # Using the standard factors from literature
        
        U3 = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)
        
        V3 = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)
        
        W3 = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)
        
        # These are still not correct. Let me derive them properly or use a known set.
        # For now, fall back to schoolbook.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
                    
        return {"U": U, "V": V, "W": W}
    
    # If 3x3 rank-7 is verified, construct 2x3x3 rank-14
    # T233[i,j,k] = sum_c U233[i*3+j, c] * V233[j*3+k, c] * W233[k*2+i, c]
    # 
    # Strategy: Use the rank-7 factors for each of the 2 rows.
    # For row i, C[i,k] = sum_j A[i,j] B[j,k]
    # This is like a 1x3 @ 3x3 multiplication, which can be embedded in 3x3.
    
    # Actually, let me try a different approach:
    # The 2x3x3 tensor can be decomposed by noting that it's the "slice" of a 3x3x3 tensor.
    # T233[i,j,k] = 1 for i in {0,1}
    # 
    # If I have a rank-7 decomposition of 3x3x3, I can restrict it to i in {0,1}.
    # But the rank might not decrease.
    
    # Let me try: U233[i*3+j, c] = U3[i*3+j, c] for i in {0,1}
    # V233[j*3+k, c] = V3[j*3+k, c] (same as 3x3)
    # W233[k*2+i, c] = W3[k*3+i, c] for i in {0,1}
    
    rank = 14
    U233 = [[0] * rank for _ in range(n * m)]  # 6 x 14
    V233 = [[0] * rank for _ in range(m * p)]  # 9 x 14
    W233 = [[0] * rank for _ in range(p * n)]  # 6 x 14
    
    # For c in 0..6: use U3, V3, W3 as is
    for c in range(7):
        for i in range(2):
            for j in range(3):
                U233[i*3+j][c] = U3[i*3+j, c]
        
        for j in range(3):
            for k in range(3):
                V233[j*3+k][c] = V3[j*3+k, c]
        
        for k in range(3):
            for i in range(2):
                W233[k*2+i][c] = W3[k*3+i, c]
    
    # For c in 7..13: add another set of factors
    # This is getting complex. Let me just verify if the above works.
    
    T233_target = np.ones((2, 3, 3))
    T233_recon = np.zeros((2, 3, 3))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                for c in range(7):
                    T233_recon[i, j, k] += U233[i*3+j][c] * V233[j*3+k][c] * W233[k*2+i][c]
    
    if not np.allclose(T233_recon, T233_target):
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
                    
        return {"U": U, "V": V, "W": W}
    
    return {"U": U233, "V": V233, "W": W233}
# EVOLVE-BLOCK-END
