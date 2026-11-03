"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Kronecker product structure
    - 4x4x6 = (2x2x2) ⊗ (2x2x3)
    - 2x2x2 uses Strassen (rank 7)
    - 2x2x3 uses Strassen for first 2 cols (rank 7) + trivial for 3rd col (rank 4) = rank 11
    - Total rank = 7 * 11 = 77
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140.0
    
    # --- 1. Define Strassen 2x2x2 decomposition ---
    # A (2x2) * B (2x2) = C (2x2)
    # U rows: i*2+j, V rows: j*2+k, W rows: k*2+i
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U1[0,0] = 1; U1[3,0] = 1
    V1[0,0] = 1; V1[3,0] = 1
    W1[0,0] = 1; W1[3,0] = 1
    
    # M2 = (a10+a11)b00
    U1[2,1] = 1; U1[3,1] = 1
    V1[0,1] = 1
    W1[1,1] = 1; W1[3,1] = -1
    
    # M3 = a00(b01-b11)
    U1[0,2] = 1
    V1[1,2] = 1; V1[3,2] = -1
    W1[2,2] = 1; W1[3,2] = 1
    
    # M4 = a11(b10-b00)
    U1[3,3] = 1
    V1[2,3] = 1; V1[0,3] = -1
    W1[0,3] = 1; W1[1,3] = 1
    
    # M5 = (a00+a01)b11
    U1[0,4] = 1; U1[1,4] = 1
    V1[3,4] = 1
    W1[0,4] = -1; W1[2,4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U1[2,5] = 1; U1[0,5] = -1
    V1[0,5] = 1; V1[1,5] = 1
    W1[3,5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U1[1,6] = 1; U1[3,6] = -1
    V1[2,6] = 1; V1[3,6] = 1
    W1[0,6] = 1

    # --- 2. Define 2x2x3 decomposition ---
    # A (2x2) * B (2x3) = C (2x3)
    # Rank 11: 7 terms for cols 0-1 (Strassen), 4 terms for col 2 (Schoolbook)
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    # Terms 0-6: Strassen for B[:, 0:2]
    # Map V1 (2x2) to V2 (2x3) for cols 0,1
    # V1 row j*2+k -> V2 row j*3+k (for k=0,1)
    # V1[0] (j=0,k=0) -> V2[0]
    # V1[1] (j=0,k=1) -> V2[1]
    # V1[2] (j=1,k=0) -> V2[3]
    # V1[3] (j=1,k=1) -> V2[4]
    
    for c in range(7):
        U2[:, c] = U1[:, c]
        
        # V mapping
        V2[0, c] = V1[0, c]
        V2[1, c] = V1[1, c]
        V2[3, c] = V1[2, c]
        V2[4, c] = V1[3, c]
        V2[2, c] = 0 # k=2 is not used in Strassen part
        V2[5, c] = 0
        
        # W mapping
        # W1 row k*2+i -> W2 row k*2+i (for k=0,1)
        W2[0, c] = W1[0, c]
        W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]
        W2[3, c] = W1[3, c]
        W2[4, c] = 0 # k=2
        W2[5, c] = 0
    
    # Terms 7-10: Schoolbook for B[:, 2]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 7: A[0,0] * B[0,2] -> C[0,2]
    U2[0, 7] = 1
    V2[2, 7] = 1  # j=0, k=2 -> 0*3+2 = 2
    W2[4, 7] = 1  # k=2, i=0 -> 2*2+0 = 4
    
    # Term 8: A[0,1] * B[1,2] -> C[0,2]
    U2[1, 8] = 1
    V2[5, 8] = 1  # j=1, k=2 -> 1*3+2 = 5
    W2[4, 8] = 1  # k=2, i=0 -> 4
    
    # Term 9: A[1,0] * B[0,2] -> C[1,2]
    U2[2, 9] = 1
    V2[2, 9] = 1
    W2[5, 9] = 1  # k=2, i=1 -> 2*2+1 = 5
    
    # Term 10: A[1,1] * B[1,2] -> C[1,2]
    U2[3, 10] = 1
    V2[5, 10] = 1
    W2[5, 10] = 1

    # --- 3. Construct full 4x4x6 decomposition via Kronecker Product ---
    # Rank = 7 * 11 = 77
    rank = 77
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    # Indices:
    # A[i,j] with i=i1*2+i2, j=j1*2+j2
    # B[j,k] with j=j1*2+j2, k=k1*3+k2
    # C[i,k] with i=i1*2+i2, k=k1*3+k2
    
    # U_full row: i*4+j = (i1*2+i2)*4 + (j1*2+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # U1 row: i1*2+j1
    # U2 row: i2*2+j2
    
    # V_full row: j*6+k = (j1*2+j2)*6 + (k1*3+k2) = 12*j1 + 6*j2 + 3*k1 + k2
    # V1 row: j1*2+k1
    # V2 row: j2*3+k2
    
    # W_full row: k*4+i = (k1*3+k2)*4 + (i1*2+i2) = 12*k1 + 4*k2 + 2*i1 + i2
    # W1 row: k1*2+i1
    # W2 row: k2*2+i2
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            
            # Fill U
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1, c1]
                            u2 = U2[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = 8*i1 + 4*i2 + 2*j1 + j2
                                U[row, c] = u1 * u2
            
            # Fill V
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1, c1]
                            v2 = V2[j2*3+k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = 12*j1 + 6*j2 + 3*k1 + k2
                                V[row, c] = v1 * v2
            
            # Fill W
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1, c1]
                            w2 = W2[k2*2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = 12*k1 + 4*k2 + 2*i1 + i2
                                W[row, c] = w1 * w2

    # --- 4. Verify ---
    def _verify(U_mat, V_mat, W_mat, n, m, p):
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
            beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        
        return np.allclose(C_computed, C_expected, atol=1e-10)

    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # --- 5. Attempt Rank Reduction (Optional, time-bounded) ---
    # Given the structure, linear dependencies are unlikely, but we can try a fast check
    # on a few random subsets or just return the 77 rank solution.
    # To be safe and fast, we return the 77 rank solution immediately.
    # If we have time, we could try to remove terms, but 77 is already an improvement over 81.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
