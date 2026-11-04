"""
Rank-189 decomposition of 6x6 matrix multiplication via Kronecker product
of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7).

Construction:
- Partition 6x6 matrices into 3x3 blocks of 2x2
- Block-level multiplication: 3x3 schoolbook (27 terms)
- Each 2x2 block product: Strassen (7 terms)
- Total rank: 27 * 7 = 189

Indexing conventions:
- U[i*m + j, r] for A[i,j]
- V[j*p + k, r] for B[j,k]
- W[k*n + i, r] for C[i,k] (trace convention)
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build 2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1: (x11+x22)(y11+y22) -> c11+=1, c22+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2: (x21+x22)(y11) -> c21+=1, c22+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # P3: (x11)(y12-y22) -> c12+=1, c22+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # P4: (x22)(y21-y11) -> c11+=1, c21+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # P5: (x11+x12)(y22) -> c11+=-1, c12+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # P6: (x21-x11)(y11+y12) -> c22+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7: (x12-x22)(y21+y22) -> c11+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_schoolbook_3x3():
    """Build 3x3 schoolbook decomposition (rank 27)."""
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
    
    return U, V, W


def _kron_decompose(U1, V1, W1, a, U2, V2, W2, b):
    """
    Kronecker product of two decompositions.
    U1,V1,W1 decompose size-a matmul; U2,V2,W2 decompose size-b matmul.
    Returns decomposition of size-(a*b) matmul.
    
    Row mapping for size-(a*b):
    - U: (i1*a + i2) * (a*b) + (j1*b + j2) where i1,j1 in [0,a), i2,j2 in [0,b)
    - V: (j1*a + j2) * (a*b) + (k1*b + k2)
    - W: (k1*a + k2) * (a*b) + (i1*b + i2)
    """
    n = a * b
    rank = U1.shape[1] * U2.shape[1]
    
    U = np.zeros((n * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * n, rank), dtype=float)
    
    # Precompute nonzero entries for efficiency
    U1_nz = [(r, c, U1[r, c]) for c in range(U1.shape[1]) for r in range(U1.shape[0]) if U1[r, c] != 0]
    V1_nz = [(r, c, V1[r, c]) for c in range(V1.shape[1]) for r in range(V1.shape[0]) if V1[r, c] != 0]
    W1_nz = [(r, c, W1[r, c]) for c in range(W1.shape[1]) for r in range(W1.shape[0]) if W1[r, c] != 0]
    U2_nz = [(r, c, U2[r, c]) for c in range(U2.shape[1]) for r in range(U2.shape[0]) if U2[r, c] != 0]
    V2_nz = [(r, c, V2[r, c]) for c in range(V2.shape[1]) for r in range(V2.shape[0]) if V2[r, c] != 0]
    W2_nz = [(r, c, W2[r, c]) for c in range(W2.shape[1]) for r in range(W2.shape[0]) if W2[r, c] != 0]
    
    # Group by column for faster lookup
    U1_by_col = {c: [(r, val) for r, col, val in U1_nz if col == c] for c in range(U1.shape[1])}
    V1_by_col = {c: [(r, val) for r, col, val in V1_nz if col == c] for c in range(V1.shape[1])}
    W1_by_col = {c: [(r, val) for r, col, val in W1_nz if col == c] for c in range(W1.shape[1])}
    U2_by_col = {c: [(r, val) for r, col, val in U2_nz if col == c] for c in range(U2.shape[1])}
    V2_by_col = {c: [(r, val) for r, col, val in V2_nz if col == c] for c in range(V2.shape[1])}
    W2_by_col = {c: [(r, val) for r, col, val in W2_nz if col == c] for c in range(W2.shape[1])}
    
    for c1 in range(U1.shape[1]):
        for c2 in range(U2.shape[1]):
            c_final = c1 * U2.shape[1] + c2
            
            # U: row = (i1*a + i2) * n + (j1*b + j2)
            # U1 row i1*a+j1, U2 row i2*b+j2
            for i1, j1, val1 in [(r // a, r % a, v) for r, v in U1_by_col[c1]]:
                for i2, j2, val2 in [(r // b, r % b, v) for r, v in U2_by_col[c2]]:
                    target_row = (i1 * b + i2) * n + (j1 * b + j2)
                    U[target_row, c_final] = val1 * val2
            
            # V: row = (j1*a + j2) * n + (k1*b + k2)
            # V1 row j1*a+k1, V2 row j2*b+k2
            for j1, k1, val1 in [(r // a, r % a, v) for r, v in V1_by_col[c1]]:
                for j2, k2, val2 in [(r // b, r % b, v) for r, v in V2_by_col[c2]]:
                    target_row = (j1 * b + j2) * n + (k1 * b + k2)
                    V[target_row, c_final] = val1 * val2
            
            # W: row = (k1*a + k2) * n + (i1*b + i2)
            # W1 row k1*a+i1, W2 row k2*b+i2
            for k1, i1, val1 in [(r // a, r % a, v) for r, v in W1_by_col[c1]]:
                for k2, i2, val2 in [(r // b, r % b, v) for r, v in W2_by_col[c2]]:
                    target_row = (k1 * b + k2) * n + (i1 * b + i2)
                    W[target_row, c_final] = val1 * val2
    
    return U, V, W


def construct():
    """
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using Kronecker product of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Build base decompositions
    U22, V22, W22 = _build_strassen_2x2()
    U33, V33, W33 = _build_schoolbook_3x3()
    
    # Kronecker product: 3x3 schoolbook ⊗ 2x2 Strassen = 6x6
    U66, V66, W66 = _kron_decompose(U33, V33, W33, 3, U22, V22, W22, 2)
    
    rank = U66.shape[1]  # Should be 189
    
    # Verify correctness with a fast tensor contraction check
    if time.time() < deadline - 5:
        np.random.seed(12345)
        A_test = np.random.randint(-3, 4, size=(n, m))
        B_test = np.random.randint(-3, 4, size=(m, p))
        C_expected = A_test @ B_test
        
        # Fast verification: compute C[i,k] = sum_j sum_r A[i,j]*U[i*m+j,r] * B[j,k]*V[j*p+k,r] * W[k*n+i,r]
        # Vectorize: for each r, compute (A * U_r) @ (B * V_r) * W_r and sum
        C_computed = np.zeros((n, p))
        for r in range(rank):
            if not (np.any(U66[:, r]) and np.any(V66[:, r]) and np.any(W66[:, r])):
                continue
            # A * U_r: element-wise multiply then reshape
            AU = (A_test * U66.reshape(n, m, rank)[:, :, r]).reshape(n, m)
            BV = (B_test * V66.reshape(m, p, rank)[:, :, r]).reshape(m, p)
            AB = AU @ BV
            C_computed += W66.reshape(p, n, rank)[:, :, r] * AB
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_fb = np.zeros((n * m, rank_school), dtype=float)
            V_fb = np.zeros((m * p, rank_school), dtype=float)
            W_fb = np.zeros((p * n, rank_school), dtype=float)
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, column] = 1
                        V_fb[j * p + k, column] = 1
                        W_fb[k * n + i, column] = 1
                        column += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}
# EVOLVE-BLOCK-END
