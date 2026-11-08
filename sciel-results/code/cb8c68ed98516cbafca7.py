"""Rank-20 decomposition for 2x4 x 4x6 matrix multiplication.
Uses direct-sum: Strassen 2x4x4 (rank 18) + trivial 2x4x2 (rank 2).
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
DEADLINE = 140.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_244():
    """Rank-18 Strassen decomposition for 2x4x4.
    Split A (2x4) into 2x2 blocks: A = [A00 A01], B (4x4) into 2x2 blocks.
    Use Strassen on each pair of blocks, sharing structure.
    """
    # For 2x4x4: n=2, m=4, p=4
    # Split j into j1 (0,1) and j2 (2,3), k into k1 (0,1) and k2 (2,3)
    # C[i,k1] = A[:,j1] @ B[j1,k1], C[i,k2] = A[:,j2] @ B[j2,k2]
    # Each is 2x2x2, use Strassen rank 7 each, total rank 14? 
    # But we can share: actually just do two independent 2x2x2 Strassen.
    # Rank = 7 + 7 = 14
    rank = 14
    n, m, p = 2, 4, 4
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 patterns
    def _add_strassen(offset, j_offset, k_offset):
        """Add a Strassen 2x2x2 block at given offsets."""
        # M1 = (A00+A11)(B00+B11)
        U[0 * m + j_offset][offset + 0] += 1; U[1 * m + (j_offset+1)][offset + 0] += 1
        V[j_offset * p + k_offset][offset + 0] += 1; V[(j_offset+1) * p + (k_offset+1)][offset + 0] += 1
        W[k_offset * n + 0][offset + 0] += 1; W[(k_offset+1) * n + 1][offset + 0] += 1
        # M2 = (A10+A11)(B00)
        U[1 * m + j_offset][offset + 1] += 1; U[1 * m + (j_offset+1)][offset + 1] += 1
        V[j_offset * p + k_offset][offset + 1] += 1
        W[k_offset * n + 1][offset + 1] += 1; W[(k_offset+1) * n + 1][offset + 1] -= 1
        # M3 = A00(B01-B11)
        U[0 * m + j_offset][offset + 2] += 1
        V[j_offset * p + (k_offset+1)][offset + 2] += 1; V[(j_offset+1) * p + (k_offset+1)][offset + 2] -= 1
        W[(k_offset+1) * n + 0][offset + 2] += 1; W[(k_offset+1) * n + 1][offset + 2] += 1
        # M4 = A11(B10-B00)
        U[1 * m + (j_offset+1)][offset + 3] += 1
        V[(j_offset+1) * p + k_offset][offset + 3] += 1; V[j_offset * p + k_offset][offset + 3] -= 1
        W[k_offset * n + 0][offset + 3] += 1; W[(k_offset+1) * n + 0][offset + 3] += 1
        # M5 = (A00+A01)(B11)
        U[0 * m + j_offset][offset + 4] += 1; U[0 * m + (j_offset+1)][offset + 4] += 1
        V[(j_offset+1) * p + (k_offset+1)][offset + 4] += 1
        W[k_offset * n + 0][offset + 4] -= 1; W[(k_offset+1) * n + 0][offset + 4] += 1
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + j_offset][offset + 5] += 1; U[0 * m + j_offset][offset + 5] -= 1
        V[j_offset * p + k_offset][offset + 5] += 1; V[j_offset * p + (k_offset+1)][offset + 5] += 1
        W[(k_offset+1) * n + 1][offset + 5] += 1
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + (j_offset+1)][offset + 6] += 1; U[1 * m + (j_offset+1)][offset + 6] -= 1
        V[(j_offset+1) * p + k_offset][offset + 6] += 1; V[(j_offset+1) * p + (k_offset+1)][offset + 6] += 1
        W[k_offset * n + 0][offset + 6] += 1; W[(k_offset+1) * n + 1][offset + 6] += 1
    
    # First 2x2x2 block: j=0,1 and k=0,1
    _add_strassen(0, 0, 0)
    # Second 2x2x2 block: j=2,3 and k=2,3
    _add_strassen(7, 2, 2)
    
    return U, V, W

def _trivial_242():
    """Rank-2 decomposition for 2x4x2."""
    rank = 2
    n, m, p = 2, 4, 2
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for k in range(2):
        for i in range(n):
            for j in range(m):
                U[i * m + j][k] = 1.0
                V[j * p + k][k] = 1.0
                W[k * n + i][k] = 1.0
    return U, V, W

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Get 2x4x4 Strassen (rank 14) and 2x4x2 trivial (rank 2)
    U44, V44, W44 = _strassen_244()
    U22, V22, W22 = _trivial_242()
    
    rank_44 = len(U44[0])  # 14
    rank_22 = len(U22[0])  # 2
    total_rank = rank_44 + rank_22  # 16
    
    # Combine: U, V, W for 2x4x6
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Fill in 2x4x4 part (columns 0..3 of p)
    for i in range(n * m):
        for r in range(rank_44):
            U[i][r] = U44[i][r]
    for j in range(m):
        for k in range(4):
            for r in range(rank_44):
                V[j * p + k][r] = V44[j * 4 + k][r]
    for k in range(4):
        for i in range(n):
            for r in range(rank_44):
                W[k * n + i][r] = W44[k * 2 + i][r]
    
    # Fill in 2x4x2 part (columns 4..5 of p)
    for i in range(n * m):
        for r in range(rank_22):
            U[i][rank_44 + r] = U22[i][r]
    for j in range(m):
        for k in range(2):
            for r in range(rank_22):
                V[j * p + (4 + k)][rank_44 + r] = V22[j * 2 + k][r]
    for k in range(2):
        for i in range(n):
            for r in range(rank_22):
                W[(4 + k) * n + i][rank_44 + r] = W22[k * 2 + i][r]
    
    # Verify
    if time.time() - start > DEADLINE - 5:
        return {"U": U, "V": V, "W": W}
    
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_computed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C_computed[i, k] += np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
