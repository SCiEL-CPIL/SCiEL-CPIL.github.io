"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check dimensions
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Test with random matrices
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        C_comp = np.zeros((n, p))
        for r in range(rank):
            u_val = U[:, r] @ A.flatten()  # sum over i,j of U[i*m+j,r]*A[i,j]
            v_val = V[:, r] @ B.flatten()  # sum over j,k of V[j*p+k,r]*B[j,k]
            C_comp += u_val * v_val * W[:, r].reshape(p, n).T
        if not np.allclose(C_comp, C_true, atol=tol):
            return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for Strassen's rank-7 decomposition of [2,2,2]."""
    # U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    # Indices: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k]
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M4 + M6
    # C10 = M2 + M3 - M6 - M7
    # C11 = M1 - M3 + M2 + M4
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00 + A11
    V[0][0] = 1; V[3][0] = 1  # B00 + B11
    W[0][0] = 1  # C00 += M1
    W[3][0] = 1  # C11 += M1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10 + A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 += M2
    W[3][1] = 1  # C11 += M2
    
    # M3 = A00(B10-B11)
    U[0][2] = 1  # A00
    V[2][2] = 1; V[3][2] = -1  # B10 - B11
    W[2][2] = 1  # C10 += M3
    W[3][2] = -1  # C11 -= M3
    
    # M4 = A11(B01-B10)
    U[3][3] = 1  # A11
    V[1][3] = 1; V[2][3] = -1  # B01 - B10
    W[0][3] = -1  # C00 -= M4
    W[1][3] = 1  # C01 += M4
    W[3][3] = 1  # C11 += M4
    
    # M5 = (A00+A10)B11
    U[0][4] = 1; U[2][4] = 1  # A00 + A10
    V[3][4] = 1  # B11
    W[0][4] = 1  # C00 += M5
    W[3][4] = 0  # C11: no M5 term
    
    # M6 = (A01-A00)(B00+B10)
    U[1][5] = 1; U[0][5] = -1  # A01 - A00
    V[0][5] = 1; V[2][5] = 1  # B00 + B10
    W[1][5] = 1  # C01 += M6
    W[2][5] = -1  # C10 -= M6
    
    # M7 = (A10-A11)(B01+B11)
    U[2][6] = 1; U[3][6] = -1  # A10 - A11
    V[1][6] = 1; V[3][6] = 1  # B01 + B11
    W[0][6] = 1  # C00 += M7
    W[2][6] = -1  # C10 -= M7
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Split m=5 into 2+3
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x3
    # B = [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    #
    # A1*B1: [2,2,6] - use Strassen for each 2x2x2 block (split 6 into 2+2+2)
    #   Each block: rank 7, total: 3*7 = 21
    # A2*B2: [2,3,6] - schoolbook: rank 36
    # Total rank: 21 + 36 = 57
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]  # 10 x 57
    V = [[0] * rank for _ in range(m * p)]  # 30 x 57
    W = [[0] * rank for _ in range(p * n)]  # 12 x 57
    
    col = 0
    
    # Part 1: A1 * B1 where A1 is 2x2 (columns 0,1 of A), B1 is 2x6 (rows 0,1 of B)
    # Split the 6 columns of B1 into 3 blocks of 2
    for block in range(3):
        k_start = block * 2
        # Get Strassen decomposition for this 2x2x2 block
        sU, sV, sW = _strassen_2x2x2()
        for r in range(7):
            # U: A1[i,j] is at position i*5+j in full U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = sU[i * 2 + j][r]
            # V: B1[j,k] is at position j*6+k in full V
            for j in range(2):
                for k in range(2):
                    V[j * p + (k_start + k)][col] = sV[j * 2 + k][r]
            # W: C[i,k] is at position k*2+i in full W
            for i in range(2):
                for k in range(2):
                    W[(k_start + k) * n + i][col] = sW[k * 2 + i][r]
            col += 1
    
    # Part 2: A2 * B2 where A2 is 2x3 (columns 2,3,4 of A), B2 is 3x6 (rows 2,3,4 of B)
    # Schoolbook: 2*3*6 = 36 terms
    for i in range(n):
        for j in range(3):
            for k in range(p):
                U[i * m + (j + 2)][col] = 1
                V[(j + 2) * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U2 = [[0] * (n*m*p) for _ in range(n * m)]
        V2 = [[0] * (n*m*p) for _ in range(m * p)]
        W2 = [[0] * (n*m*p) for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][c] = 1
                    V2[j * p + k][c] = 1
                    W2[k * n + i][c] = 1
                    c += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
