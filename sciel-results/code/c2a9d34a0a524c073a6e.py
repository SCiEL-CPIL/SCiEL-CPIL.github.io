"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition(n, m, p):
    """
    Builds a rank-77 decomposition for 4x4 x 4x6 matrix multiplication.
    Uses tensor product of Strassen 2x2x2 (rank 7) and 2x2x3 (rank 11).
    2x2x3 is Strassen 2x2x2 (rank 7) + Schoolbook 2x2x1 (rank 4).
    """
    # 1. Strassen 2x2x2 terms
    # U, V, W are 4x7 matrices (flattened 2x2)
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a10+a11)b00
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a00(b01-b11)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a11(b10-b00)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a00+a01)b11
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1

    # 2. 2x2x3 decomposition
    # Rank 11: 7 from Strassen (cols 0,1) + 4 from Schoolbook (col 2)
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float) # 2x3 matrix flattened: 2*3=6
    W2 = np.zeros((6, 11), dtype=float) # 2x3 matrix flattened: 2*3=6
    
    # Copy Strassen part to first 7 columns
    # V1 is 2x2 (rows 0,1,2,3 for j*2+k). V2 is 2x3 (rows 0..5 for j*3+k).
    # j=0,k=0 -> V2[0]; j=0,k=1 -> V2[1]; j=1,k=0 -> V2[3]; j=1,k=1 -> V2[4].
    for c in range(7):
        U2[:, c] = U1[:, c]
        V2[0, c] = V1[0, c]
        V2[1, c] = V1[1, c]
        V2[3, c] = V1[2, c]
        V2[4, c] = V1[3, c]
        # V2[2] and V2[5] correspond to k=2, which is 0 in Strassen part
        
        # W1 is 2x2 (rows 0..3 for k*2+i). W2 is 2x3 (rows 0..5 for k*2+i).
        # k=0,i=0 -> W2[0]; k=0,i=1 -> W2[1]; k=1,i=0 -> W2[2]; k=1,i=1 -> W2[3].
        W2[0, c] = W1[0, c]
        W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]
        W2[3, c] = W1[3, c]
        # W2[4] and W2[5] correspond to k=2, which is 0 in Strassen part

    # Schoolbook part for 3rd column (k=2)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 7: A[0,0]B[0,2] -> C[0,2]
    U2[0, 7] = 1
    V2[2, 7] = 1 # j=0, k=2 -> 0*3+2=2
    W2[4, 7] = 1 # k=2, i=0 -> 2*2+0=4
    
    # Term 8: A[0,1]B[1,2] -> C[0,2]
    U2[1, 8] = 1
    V2[5, 8] = 1 # j=1, k=2 -> 1*3+2=5
    W2[4, 8] = 1 # k=2, i=0 -> 2*2+0=4
    
    # Term 9: A[1,0]B[0,2] -> C[1,2]
    U2[2, 9] = 1
    V2[2, 9] = 1 # j=0, k=2 -> 0*3+2=2
    W2[5, 9] = 1 # k=2, i=1 -> 2*2+1=5
    
    # Term 10: A[1,1]B[1,2] -> C[1,2]
    U2[3, 10] = 1
    V2[5, 10] = 1 # j=1, k=2 -> 1*3+2=5
    W2[5, 10] = 1 # k=2, i=1 -> 2*2+1=5

    # 3. Kronecker Product
    # Total rank = 7 * 11 = 77
    rank = 77
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((24, rank), dtype=float)
    W = np.zeros((24, rank), dtype=float)
    
    # Indices:
    # A (4x4): i=2*i1+i2, j=2*j1+j2. Row = i*4+j
    # B (4x6): j=2*j1+j2, k=3*k1+k2. Row = j*6+k
    # C (4x6): i=2*i1+i2, k=3*k1+k2. Row = k*4+i
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            
            # U: (i1,j1) from U1, (i2,j2) from U2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U1[i1*2+j1, c1]
                    if u1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U2[i2*2+j2, c2]
                            if u2 == 0: continue
                            row = (i1*2+i2)*4 + (j1*2+j2)
                            U[row, c] = u1 * u2
            
            # V: (j1,k1) from V1, (j2,k2) from V2
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V1[j1*2+k1, c1]
                    if v1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v2 = V2[j2*3+k2, c2]
                            if v2 == 0: continue
                            row = (j1*2+j2)*6 + (k1*3+k2)
                            V[row, c] = v1 * v2
            
            # W: (k1,i1) from W1, (k2,i2) from W2
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W1[k1*2+i1, c1]
                    if w1 == 0: continue
                    for k2 in range(3):
                        for i2 in range(2):
                            w2 = W2[k2*2+i2, c2]
                            if w2 == 0: continue
                            row = (k1*3+k2)*4 + (i1*2+i2)
                            W[row, c] = w1 * w2

    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0
    
    U, V, W = _build_decomposition(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
