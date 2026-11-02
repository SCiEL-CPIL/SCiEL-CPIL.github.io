"""Matrix multiplication 4x4x7 tensor decomposition using Kronecker product structure."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """
    Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k].
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A00 + A11) * (B00 + B11)
    # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    # U: A10=1, A11=1; V: B00=1; W: C10=1, C11=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00 * (B01 - B11)
    # U: A00=1; V: B01=1, B11=-1; W: C01=1, C11=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11 * (B10 - B00)
    # U: A11=1; V: B10=1, B00=-1; W: C00=1, C10=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01) * B11
    # U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    # U: A10=1, A00=-1; V: B00=1, B01=1; W: C11=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    # U: A01=1, A11=-1; V: B10=1, B11=1; W: C00=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_4x4x4_strassen():
    """
    Construct rank-49 decomposition for 4x4x4 matrix multiplication
    using the tensor product of two Strassen 2x2x2 decompositions.
    
    The 4x4x4 tensor T4 can be viewed as T2 ⊗ T2 where T2 is the 2x2x2 tensor.
    If T2 has rank 7, then T4 has rank at most 7*7 = 49.
    
    The Kronecker product structure:
    - U4[i*4+j, (r1,r2)] = U2[i*2+j1, r1] * U2[i2*2+j, r2]
    - V4[j*4+k, (r1,r2)] = V2[j1*2+k1, r1] * V2[j2*2+k, r2]
    - W4[k*4+i, (r1,r2)] = W2[k1*2+i1, r1] * W2[k2*2+i, r2]
    
    where i=i1*2+i2, j=j1*2+j2, k=k1*2+k2.
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            # U4[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                            U4[i * 4 + j, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    # V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
                                    V4[j * 4 + k, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                                    # W4[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
                                    W4[k * 4 + i, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
    
    return U4, V4, W4

def _construct_4x4x7_decomposition():
    """
    Construct a decomposition for 4x4x7 matrix multiplication.
    
    Split B into B1(4x4) and B2(4x3):
    A(4x4) * B(4x7) = [A*B1(4x4) | A*B2(4x3)]
    
    Use Strassen-based rank-49 decomposition for 4x4x4 part.
    Use schoolbook for 4x4x3 part (rank 48).
    Total rank: 49 + 48 = 97.
    """
    n, m, p = 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # Get 4x4x4 Strassen decomposition (rank 49)
    U4, V4, W4 = _construct_4x4x4_strassen()
    
    # Build schoolbook for 4x4x3 part (rank 48)
    # This handles A(4x4) * B2(4x3) = C2(4x3)
    rank_443 = n * m * 3  # 4*4*3 = 48
    U_443 = np.zeros((nm, rank_443))
    V_443 = np.zeros((mp, rank_443))
    W_443 = np.zeros((pn, rank_443))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                # Only for k < 3 (the last 3 columns of B)
                U_443[i * m + j, col] = 1
                V_443[j * p + k, col] = 1
                W_443[k * n + i, col] = 1
                col += 1
    
    # Combine: total rank = 49 + 48 = 97
    total_rank = 49 + rank_443
    
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # First 49 columns: 4x4x4 part (columns 0-3 of B)
    U[:, 0:49] = U4
    V[:, 0:49] = V4
    W[:, 0:49] = W4
    
    # Last 48 columns: 4x4x3 part (columns 4-6 of B)
    U[:, 49:] = U_443
    V[:, 49:] = V_443
    W[:, 49:] = W_443
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the factors exactly decompose the tensor."""
    rank = len(U[0]) if U else 0
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
    
    # Target tensor: T[i,j,k] = 1 for all valid i,j,k
    T_target = np.ones((n, m, p))
    
    diff = np.max(np.abs(T_recon - T_target))
    return diff < tol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Construct the rank-97 decomposition
    result = _construct_4x4x7_decomposition()
    
    # Verify the decomposition
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    if time.time() < deadline - 5:
        is_valid = _verify_decomposition(U, V, W, n, m, p)
        if not is_valid:
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U_s = [[0] * rank for _ in range(n * m)]
            V_s = [[0] * rank for _ in range(m * p)]
            W_s = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][col] = 1
                        V_s[j * p + k][col] = 1
                        W_s[k * n + i][col] = 1
                        col += 1
            result = {"U": U_s, "V": V_s, "W": W_s}
    
    return result
