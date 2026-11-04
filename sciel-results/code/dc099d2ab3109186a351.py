"""Winograd-style structured decomposition for 2x3 * 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _validate(U, V, W):
    """Check that the decomposition exactly reproduces the matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    A = rng.rand(n, m)
    B = rng.rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        Mc = U[:n*m, c].reshape(n, m)
        Nc = V[:m*p, c].reshape(m, p)
        wc = W[:p*n, c].reshape(p, n).T
        inner = Mc @ Nc
        C_computed += wc * inner
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Rank-17 decomposition
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: Strassen 2x2x2 for A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # Indices: A[i, j] -> U[i*3+j], B[j, k] -> V[j*3+k], C[i, k] -> W[k*2+i]
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0][col]=1; U[4][col]=1
    V[0][col]=1; V[4][col]=1
    W[0][col]=1; W[3][col]=1
    col+=1
    
    # M2: (A10+A11)B00 -> C10, C11
    U[3][col]=1; U[4][col]=1
    V[0][col]=1
    W[2][col]=1; W[3][col]=1
    col+=1
    
    # M3: A00(B01-B11) -> C00, C01
    U[0][col]=1
    V[1][col]=1; V[4][col]=-1
    W[0][col]=1; W[1][col]=1
    col+=1
    
    # M4: A11(B10-B00) -> C10, C11
    U[4][col]=1
    V[3][col]=1; V[0][col]=-1
    W[2][col]=1; W[3][col]=1
    col+=1
    
    # M5: (A00+A01)B11 -> C00, C01
    U[0][col]=1; U[1][col]=1
    V[4][col]=1
    W[0][col]=1; W[1][col]=1
    col+=1
    
    # M6: (A10-A00)(B00+B01) -> C10 (-1), C01 (-1)
    U[3][col]=1; U[0][col]=-1
    V[0][col]=1; V[1][col]=1
    W[2][col]=-1; W[1][col]=-1
    col+=1
    
    # M7: (A01-A11)(B10+B11) -> C01 (-1), C10 (1)
    U[1][col]=1; U[4][col]=-1
    V[3][col]=1; V[4][col]=1
    W[1][col]=-1; W[2][col]=1
    col+=1
    
    # Block 2: A[:, 0:2] * B[0:2, 2] -> C[:, 2]
    # 4 terms
    # A00 B02 -> C02
    U[0][col]=1; V[2][col]=1; W[4][col]=1
    col+=1
    # A01 B12 -> C02
    U[1][col]=1; V[5][col]=1; W[4][col]=1
    col+=1
    # A10 B02 -> C12
    U[3][col]=1; V[2][col]=1; W[5][col]=1
    col+=1
    # A11 B12 -> C12
    U[4][col]=1; V[5][col]=1; W[5][col]=1
    col+=1
    
    # Block 3: A[:, 2] * B[2, 0:2] -> C[:, 0:2]
    # 4 terms
    # A02 B20 -> C00
    U[2][col]=1; V[6][col]=1; W[0][col]=1
    col+=1
    # A02 B21 -> C01
    U[2][col]=1; V[7][col]=1; W[1][col]=1
    col+=1
    # A12 B20 -> C10
    U[5][col]=1; V[6][col]=1; W[2][col]=1
    col+=1
    # A12 B21 -> C11
    U[5][col]=1; V[7][col]=1; W[3][col]=1
    col+=1
    
    # Block 4: A[:, 2] * B[2, 2] -> C[:, 2]
    # 2 terms
    # A02 B22 -> C02
    U[2][col]=1; V[8][col]=1; W[4][col]=1
    col+=1
    # A12 B22 -> C12
    U[5][col]=1; V[8][col]=1; W[5][col]=1
    col+=1
    
    result = {"U": U, "V": V, "W": W}
    
    if _validate(U, V, W):
        return result
    
    # Fallback to schoolbook
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
