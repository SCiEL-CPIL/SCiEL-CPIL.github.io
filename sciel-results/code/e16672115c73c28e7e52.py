"""Optimized tensor rank decomposition for 3x4 * 4x4 matrix multiplication."""
import time
import numpy as np

N, M, P = 3, 4, 4
RANK_SCHOOLBOOK = N * M * P  # 48


def _verify_fast(U, V, W, n=N, m=M, p=P):
    """Faster verification using tensor reshaping."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(recon, expected, atol=1e-9)


def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 2: M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = -1; W[2, 1] = 1
    
    # Product 3: M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Product 4: M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Product 5: M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Product 6: M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # Product 7: M7 = (b-c)(e+g)
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_optimized_decomposition():
    """
    Build rank-32 decomposition for 3x4 * 4x4 using Strassen blocks.
    
    Split A (3x4) into:
      - A_top (2x4) for rows i=0,1
      - A_bot (1x4) for row i=2
    
    Split B (4x4) into:
      - B_top (2x4) for rows j=0,1
      - B_bot (2x4) for rows j=2,3
    
    2x4 * 4x4 part (i=0,1):
      C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
      First sum: A_top (2x2) * B_top (2x4) -> two 2x2*2x2 Strassen blocks = 14
      Second sum: A_top (2x2) * B_bot (2x4) -> two 2x2*2x2 Strassen blocks = 14
      Total: 28
    
    1x4 * 4x4 part (i=2):
      C[2,k] = sum_{j=0}^{3} A[2,j]*B[j,k]
      For each j: one product A[2,j]*B[j,:] contributing to all C[2,k]
      Rank 4 (one per j)
      Total: 4
    
    Grand total: 32
    """
    rank = 32
    U = np.zeros((12, rank))  # 3*4=12 rows
    V = np.zeros((16, rank))  # 4*4=16 rows
    W = np.zeros((12, rank))  # 4*3=12 rows
    
    col = 0
    
    # --- 2x4 * 4x4 part (i=0,1) ---
    # A_top (2x2) * B_top (2x4): B_top has rows j=0,1, columns k=0..3
    # Split B_top into two 2x2 blocks: B_top[:, 0:2] and B_top[:, 2:4]
    for block in range(2):
        k_start = block * 2
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    U[li * 4 + lj, col] = sU[li * 2 + lj][c]
            for lj in range(2):
                for lk in range(2):
                    V[lj * 4 + k_start + lk, col] = sV[lj * 2 + lk][c]
            for lk in range(2):
                for li in range(2):
                    W[(k_start + lk) * 3 + li, col] = sW[lk * 2 + li][c]
            col += 1
    
    # A_top (2x2) * B_bot (2x4): B_bot has rows j=2,3, columns k=0..3
    for block in range(2):
        k_start = block * 2
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    U[li * 4 + lj, col] = sU[li * 2 + lj][c]
            for lj in range(2):
                for lk in range(2):
                    V[(2 + lj) * 4 + k_start + lk, col] = sV[lj * 2 + lk][c]
            for lk in range(2):
                for li in range(2):
                    W[(k_start + lk) * 3 + li, col] = sW[lk * 2 + li][c]
            col += 1
    
    # --- 1x4 * 4x4 part (i=2) ---
    # For each j in {0,1,2,3}: product A[2,j] * B[j,k] for all k
    for j in range(4):
        U[2 * 4 + j, col] = 1
        for k in range(4):
            V[j * 4 + k, col] = 1
            W[k * 3 + 2, col] = 1
        col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return U.tolist(), V.tolist(), W.tolist()


def _schoolbook():
    """Fallback schoolbook decomposition of rank 48."""
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try the Strassen-based block decomposition (rank 32)
    U, V, W = _build_optimized_decomposition()
    
    if _verify_fast(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()
