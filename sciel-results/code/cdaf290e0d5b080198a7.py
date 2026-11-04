import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Helper to create Strassen 2x2x2 decomposition
    def get_strassen_2x2():
        # U, V, W shapes (4, 7)
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices: 0:11, 1:12, 2:21, 3:22
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C21 += M2
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C12 += M3
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C11 += M4, C21 += M4
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[0, 4] = -1 # C12 += M5, C11 -= M5
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C22 += M6
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C11 += M7
        
        return U, V, W

    # Helper to create standard 2x1x1 or similar
    # For 4x2x2, we use two Strassen blocks.
    def get_4x2x2():
        # Split 4 rows into two 2x2 blocks.
        # Block 1: Rows 0,1. Block 2: Rows 2,3.
        U1, V1, W1 = get_strassen_2x2()
        U2, V2, W2 = get_strassen_2x2()
        
        U = np.zeros((8, 14)) # 4*2 = 8
        V = np.zeros((4, 14)) # 2*2 = 4
        W = np.zeros((4, 14)) # 2*4 = 8? No.
        # W shape for 2x2x2 is (2*2, 7) = (4,7).
        # Here p=2, n=2. W is (p*n, rank) = (4, rank).
        
        # Combine
        U[:4, :7] = U1
        U[4:, :7] = U2
        
        V[:4, :7] = V1
        V[:4, 7:] = V2
        
        W[:4, :7] = W1
        W[4:, :7] = W2
        
        return U, V, W

    # Build 4x2x6 (Rank 42)
    # Split p=6 into three 2s.
    def get_4x2x6():
        U1, V1, W1 = get_4x2x2()
        U2, V2, W2 = get_4x2x2()
        U3, V3, W3 = get_4x2x2()
        
        U = np.zeros((8, 42))
        V = np.zeros((12, 42)) # 2*6 = 12
        W = np.zeros((8, 42)) # 6*4 = 24? No.
        # W shape for 4x2x2 is (2*4, 14) = (8, 14).
        
        U[:, :14] = U1
        U[:, 14:28] = U2
        U[:, 28:] = U3
        
        V[:, :4] = V1
        V[:, 4:8] = V2
        V[:, 8:] = V3
        
        W[:, :14] = W1
        W[:, 14:28] = W2
        W[:, 28:] = W3
        
        return U, V, W

    # Build 4x1x6 (Rank 24)
    def get_4x1x6():
        U = np.zeros((4, 24)) # 4*1 = 4
        V = np.zeros((6, 24)) # 1*6 = 6
        W = np.zeros((24, 24)) # 6*4 = 24
        
        col = 0
        for i in range(4):
            for k in range(6):
                U[i, col] = 1
                V[k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1
        return U, V, W

    # Build 4x3x6 (Rank 66)
    # Split m=3 into 2+1.
    def get_4x3x6():
        U1, V1, W1 = get_4x2x6()
        U2, V2, W2 = get_4x1x6()
        
        U = np.zeros((12, 66)) # 4*3 = 12
        V = np.zeros((18, 66)) # 3*6 = 18
        W = np.zeros((24, 66)) # 6*4 = 24
        
        U[:8, :42] = U1
        U[8:, 42:] = U2
        
        V[:12, :42] = V1
        V[12:, 42:] = V2
        
        W[:, :42] = W1
        W[:, 42:] = W2
        
        return U, V, W

    # Final 4x5x6 (Rank 108)
    # Split m=5 into 2+3.
    U1, V1, W1 = get_4x2x6()
    U2, V2, W2 = get_4x3x6()
    
    U = np.zeros((20, 108)) # 4*5 = 20
    V = np.zeros((30, 108)) # 5*6 = 30
    W = np.zeros((24, 108)) # 6*4 = 24
    
    U[:8, :42] = U1
    U[8:, 42:] = U2
    
    V[:12, :42] = V1
    V[12:, 42:] = V2
    
    W[:, :42] = W1
    W[:, 42:] = W2
    
    # Verify
    U_r = U.reshape(n, m, 108)
    V_r = V.reshape(m, p, 108)
    W_r = W.reshape(p, n, 108)
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
        V_school = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
        W_school = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1.0
                    V_school[j * p + k, col] = 1.0
                    W_school[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
