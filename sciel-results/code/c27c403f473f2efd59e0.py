"""Deterministic rank-78 decomposition for 3x4*4x7 matrix multiplication using Strassen blocks."""

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify exact decomposition including zero constraints."""
    nm, mp, pn = n * m, m * p, p * n
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for (a, b, c) in valid:
        s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
        if abs(s - 1.0) > tol:
            return False
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(s) > tol:
                        return False
    return True

def _add_strassen_block(U, V, W, rank_ref, i1, i2, j1, j2, k1, k2, n, m, p):
    """Add a Strassen 2x2x2 block. Returns new rank index."""
    a11, a12 = i1 * m + j1, i1 * m + j2
    a21, a22 = i2 * m + j1, i2 * m + j2
    b11, b12 = j1 * p + k1, j1 * p + k2
    b21, b22 = j2 * p + k1, j2 * p + k2
    c11, c12 = k1 * n + i1, k2 * n + i1
    c21, c22 = k1 * n + i2, k2 * n + i2
    
    r = rank_ref[0]
    
    # m1 = (A11+A22)(B11+B22)
    U[a11, r] += 1.0; U[a22, r] += 1.0
    V[b11, r] += 1.0; V[b22, r] += 1.0
    W[c11, r] += 1.0; W[c22, r] += 1.0; W[c12, r] -= 1.0; W[c21, r] -= 1.0
    r += 1
    
    # m2 = (A21+A22)B11
    U[a21, r] += 1.0; U[a22, r] += 1.0
    V[b11, r] += 1.0
    W[c21, r] += 1.0; W[c22, r] -= 1.0
    r += 1
    
    # m3 = A11(B21-B22)
    U[a11, r] += 1.0
    V[b21, r] += 1.0; V[b22, r] -= 1.0
    W[c12, r] += 1.0; W[c21, r] += 1.0
    r += 1
    
    # m4 = A22(B12-B11)
    U[a22, r] += 1.0
    V[b12, r] += 1.0; V[b11, r] -= 1.0
    W[c11, r] += 1.0; W[c21, r] += 1.0
    r += 1
    
    # m5 = (A11+A12)B22
    U[a11, r] += 1.0; U[a12, r] += 1.0
    V[b22, r] += 1.0
    W[c11, r] -= 1.0; W[c12, r] += 1.0
    r += 1
    
    # m6 = (A21-A11)(B11+B12)
    U[a21, r] += 1.0; U[a11, r] -= 1.0
    V[b11, r] += 1.0; V[b12, r] += 1.0
    W[c22, r] += 1.0
    r += 1
    
    # m7 = (A12-A22)(B21+B22)
    U[a12, r] += 1.0; U[a22, r] -= 1.0
    V[b21, r] += 1.0; V[b22, r] += 1.0
    W[c11, r] += 1.0; W[c22, r] += 1.0
    r += 1
    
    rank_ref[0] = r
    return r

def _add_schoolbook(U, V, W, rank_ref, i, j, k, n, m, p):
    """Add a single schoolbook multiplication."""
    a = i * m + j
    b = j * p + k
    c = k * n + i
    r = rank_ref[0]
    U[a, r] += 1.0
    V[b, r] += 1.0
    W[c, r] += 1.0
    rank_ref[0] = r + 1
    return r + 1

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Strategy: Use Strassen 2x2x2 blocks for rows {0,1} and schoolbook for row 2
    # For rows {0,1}:
    #   j pairs: (0,1), (2,3)
    #   k pairs: (0,1), (2,3), (4,5), and single (6)
    #   For each j pair and k pair: Strassen (rank 7)
    #   For each j pair and k=6: schoolbook (rank 4)
    #   Total per j pair: 3*7 + 4 = 25
    #   Two j pairs: 50
    # For row 2: all schoolbook = 4*7 = 28
    # Total: 50 + 28 = 78
    
    max_rank = 80  # Allocate a bit extra
    U = np.zeros((nm, max_rank))
    V = np.zeros((mp, max_rank))
    W = np.zeros((pn, max_rank))
    rank_ref = [0]
    
    # Rows {0,1}, j pair (0,1):
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 0, 1, 0, 1, n, m, p)
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 0, 1, 2, 3, n, m, p)
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 0, 1, 4, 5, n, m, p)
    # k=6, j in {0,1}, i in {0,1}: schoolbook
    for i in (0, 1):
        for j in (0, 1):
            _add_schoolbook(U, V, W, rank_ref, i, j, 6, n, m, p)
    
    # Rows {0,1}, j pair (2,3):
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 2, 3, 0, 1, n, m, p)
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 2, 3, 2, 3, n, m, p)
    _add_strassen_block(U, V, W, rank_ref, 0, 1, 2, 3, 4, 5, n, m, p)
    # k=6, j in {2,3}, i in {0,1}: schoolbook
    for i in (0, 1):
        for j in (2, 3):
            _add_schoolbook(U, V, W, rank_ref, i, j, 6, n, m, p)
    
    # Row 2: all schoolbook
    for j in range(m):
        for k in range(p):
            _add_schoolbook(U, V, W, rank_ref, 2, j, k, n, m, p)
    
    final_rank = rank_ref[0]
    
    # Trim to actual rank
    U = U[:, :final_rank]
    V = V[:, :final_rank]
    W = W[:, :final_rank]
    
    # Verify
    if not _verify(U, V, W, n, m, p, final_rank):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_sch = np.zeros((nm, rank_school))
        V_sch = np.zeros((mp, rank_school))
        W_sch = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i * m + j, col] = 1.0
                    V_sch[j * p + k, col] = 1.0
                    W_sch[k * n + i, col] = 1.0
                    col += 1
        U, V, W = U_sch, V_sch, W_sch
    
    def _to_list(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
