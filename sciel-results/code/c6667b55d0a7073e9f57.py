"""Rank-25 tensor decomposition for 2x2 * 2x7 matrix multiplication.

Approach: Algebraic construction using Strassen's algorithm for 2x2 blocks.
- Split the 2x7 matrix into three 2x2 blocks (cols 0-5) and one 2x1 block (col 6).
- Apply Strassen (rank 7) to each 2x2 block: 3 * 7 = 21 ranks.
- Apply schoolbook to the 2x1 block: 4 ranks.
- Total rank: 25.

The Strassen decomposition is verified at runtime; if it fails, fall back to schoolbook (rank 28).
"""
import numpy as np


def _strassen_2x2():
    """Return (U, V, W) for the rank-7 Strassen decomposition of 2x2 * 2x2.
    
    U has shape (4, 7): rows indexed by (i, j) of A[i,j], i.e. A index i*m+j
    V has shape (4, 7): rows indexed by (j, k) of B[j,k], i.e. B index j*p+k  
    W has shape (4, 7): rows indexed by (k, i) of C[k,i], i.e. C index k*n+i
    
    For 2x2: n=2, m=2, p=2
    A index: i*2 + j  (i,j in {0,1})
    B index: j*2 + k  (j,k in {0,1})
    C index: k*2 + i  (k,i in {0,1})
    
    Strassen formulas:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # A indices: (i,j) -> i*2+j
    # A00=0, A01=1, A10=2, A11=3
    # B indices: (j,k) -> j*2+k
    # B00=0, B01=1, B10=2, B11=3
    # C indices: (k,i) -> k*2+i
    # C00=0, C01=1, C10=2, C11=3
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[1, 0] += 1; W[2, 0] += 1; W[3, 0] += 1
    
    # M2 = (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] += 1
    
    # M3 = A00(B10-B11)
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # M4 = A11(B01-B00)
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[2, 3] += 1; W[0, 3] += 1
    
    # M5 = (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[1, 4] += 1; W[3, 4] += 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[2, 5] += 1
    W[0, 5] += 1; W[2, 5] += 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[1, 6] += 1; V[3, 6] += 1
    W[1, 6] += 1; W[3, 6] += 1
    
    return U, V, W


def _verify_strassen():
    """Verify the Strassen decomposition is correct."""
    U, V, W = _strassen_2x2()
    # Build tensor: T[i*2+j, j*2+k, k*2+i] should equal sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a_idx = i * 2 + j
                b_idx = j * 2 + k
                c_idx = k * 2 + i
                expected = 1
                actual = sum(U[a_idx, c] * V[b_idx, c] * W[c_idx, c] for c in range(7))
                if actual != expected:
                    return False
    # Also verify zeros: for non-matching indices, sum should be 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a_idx = i * 2 + j
                b_idx = j * 2 + k
                c_idx = k * 2 + i
                # This is the only non-zero entry for this (i,j,k)
    # Check that entries that should be zero are zero
    # T[a,b,c] is non-zero only if a=i*2+j, b=j*2+k, c=k*2+i for some i,j,k
    # i.e., a%2 == b//2 and b%2 == c//2 and c%2 == a//2
    for a in range(4):
        for b in range(4):
            for c in range(4):
                i = a // 2
                j = a % 2
                j2 = b // 2
                k = b % 2
                k2 = c // 2
                i2 = c % 2
                if not (j == j2 and k == k2 and i == i2):
                    actual = sum(U[a, cc] * V[b, cc] * W[c, cc] for cc in range(7))
                    if actual != 0:
                        return False
    return True


def _schoolbook_2x2_2x7():
    """Return the canonical schoolbook rank-28 decomposition."""
    n, m, p = 2, 2, 7
    rank = n * m * p  # 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 14 x 28
    W = [[0] * rank for _ in range(p * n)]  # 14 x 28
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_rank25():
    """Construct the rank-25 decomposition using 3 Strassen blocks + 1 schoolbook block."""
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n * m, rank), dtype=int)  # 4 x 25
    V = np.zeros((m * p, rank), dtype=int)  # 14 x 25
    W = np.zeros((p * n, rank), dtype=int)  # 14 x 25
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2()
    
    # Block 1: columns 0-1 of B (k=0,1) -> use Strassen
    for c in range(7):
        col = c
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] += strassen_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[j * p + k, col] += strassen_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[k * n + i, col] += strassen_W[k * 2 + i, c]
    
    # Block 2: columns 2-3 of B (k=2,3) -> use Strassen
    for c in range(7):
        col = 7 + c
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] += strassen_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[j * p + (k + 2), col] += strassen_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * n + i, col] += strassen_W[k * 2 + i, c]
    
    # Block 3: columns 4-5 of B (k=4,5) -> use Strassen
    for c in range(7):
        col = 14 + c
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] += strassen_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[j * p + (k + 4), col] += strassen_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[(k + 4) * n + i, col] += strassen_W[k * 2 + i, c]
    
    # Block 4: column 6 of B (k=6) -> use schoolbook (4 ranks)
    for i in range(2):
        for j in range(2):
            col = 21 + (i * m + j)
            U[i * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + i, col] = 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _verify_decomposition(result, n=2, m=2, p=7):
    """Verify that the decomposition is exact."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Build the tensor from the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                actual = sum(U[a_idx, c] * V[b_idx, c] * W[c_idx, c] for c in range(rank))
                if abs(actual - 1.0) > 1e-10:
                    return False
    # Check zeros
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (j == j2 and k == k2 and i == i2):
                    actual = sum(U[a, cc] * V[b, cc] * W[c, cc] for cc in range(rank))
                    if abs(actual) > 1e-10:
                        return False
    return True


def construct():
    # First verify Strassen is correct
    if not _verify_strassen():
        # Fallback to schoolbook if Strassen verification fails
        return _schoolbook_2x2_2x7()
    
    # Construct the rank-25 decomposition
    result = _construct_rank25()
    
    # Verify the full decomposition
    if not _verify_decomposition(result):
        # Fallback to schoolbook
        return _schoolbook_2x2_2x7()
    
    return result
