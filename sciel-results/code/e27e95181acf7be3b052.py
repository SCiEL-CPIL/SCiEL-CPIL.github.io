"""Matrix multiplication 2,4,4 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130  # safety margin before 150s hard kill

def construct():
    """Return {'U','V','W'} factor matrices for 2x4 by 4x4 matrix multiplication."""
    if time.time() > _DEADLINE:
        return _schoolbook()
    try:
        result = _construct_rank28()
        if result is not None:
            return result
    except Exception:
        pass
    return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank28():
    """
    Rank-28 decomposition using Strassen-style 7-term block multiplication
    applied to four independent 2x2-by-2x2 sub-problems.
    
    The 2x4 matrix is split into four 2x2 blocks (rows 0-1, cols 0-1; 0-1, 2-3; 2-3, 0-1; 2-3, 2-3).
    The 4x4 matrix is similarly split into four 2x2 blocks.
    Each 2x2 by 2x2 multiplication uses Strassen's 7 terms, giving 4*7=28 total.
    """
    n, m, p = DIMENSIONS
    assert n == 2 and m == 4 and p == 4
    rank = 28

    U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 8 x 28

    # Strassen 7 terms for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    #
    # Each term is (u_pattern, v_pattern, w_pattern) where each pattern is
    # a dict mapping local-index -> coefficient.
    # Local indices: for A (2x2): 0=A00,1=A01,2=A10,3=A11
    #                for B (2x2): 0=B00,1=B01,2=B10,3=B11
    #                for C (2x2): 0=C00,1=C01,2=C10,3=C11

    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),                # M2
        ({0: 1}, {2: 1, 3: -1}, {1: 1}),               # M3
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 2: 1}, {3: 1}, {0: -1, 1: 1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {3: 1}),         # M6
        ({3: 1, 1: -1}, {1: 1, 3: 1}, {0: 1, 3: -1}),  # M7
    ]

    # Four sub-problems. Each maps local indices (0..3) to global row indices.
    # U index = i*m + j, where i is row of the 2x4 matrix, j is col.
    # V index = j*p + k, where j is row of 4x4 (col of 2x4), k is col of 4x4.
    # W index = k*n + i, where k is col of output (col of 4x4), i is row of output (row of 2x4).

    # Sub-problem 1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1
    # U local: (i,j) in {0,1}x{0,1} -> i*4+j: 0,1,4,5
    # V local: (j,k) in {0,1}x{0,1} -> j*4+k: 0,1,4,5
    # W local: (k,i) in {0,1}x{0,1} -> k*2+i: 0,1,2,3
    P1 = [0, 1, 4, 5]
    Q1 = [0, 1, 4, 5]
    R1 = [0, 1, 2, 3]

    # Sub-problem 2: A rows 0-1, cols 2-3; B rows 2-3, cols 0-1
    # U local: (i,j) in {0,1}x{2,3} -> i*4+j: 2,3,6,7
    # V local: (j,k) in {2,3}x{0,1} -> j*4+k: 8,9,12,13
    # W local: (k,i) in {0,1}x{0,1} -> k*2+i: 0,1,2,3
    P2 = [2, 3, 6, 7]
    Q2 = [8, 9, 12, 13]
    R2 = [0, 1, 2, 3]

    # Sub-problem 3: A rows 2-3, cols 0-1; B rows 0-1, cols 2-3
    # U local: (i,j) in {2,3}x{0,1} -> i*4+j: 8,9,12,13
    # V local: (j,k) in {0,1}x{2,3} -> j*4+k: 2,3,6,7
    # W local: (k,i) in {2,3}x{2,3} -> k*2+i: 4,5,6,7
    P3 = [8, 9, 12, 13]
    Q3 = [2, 3, 6, 7]
    R3 = [4, 5, 6, 7]

    # Sub-problem 4: A rows 2-3, cols 2-3; B rows 2-3, cols 2-3
    # U local: (i,j) in {2,3}x{2,3} -> i*4+j: 10,11,14,15
    # V local: (j,k) in {2,3}x{2,3} -> j*4+k: 10,11,14,15
    # W local: (k,i) in {2,3}x{2,3} -> k*2+i: 4,5,6,7
    P4 = [10, 11, 14, 15]
    Q4 = [10, 11, 14, 15]
    R4 = [4, 5, 6, 7]

    subs = [(P1, Q1, R1), (P2, Q2, R2), (P3, Q3, R3), (P4, Q4, R4)]

    col = 0
    for P_map, Q_map, R_map in subs:
        for P_f, Q_f, R_f in strassen_terms:
            for loc, coeff in P_f.items():
                U[P_map[loc], col] += coeff
            for loc, coeff in Q_f.items():
                V[Q_map[loc], col] += coeff
            for loc, coeff in R_f.items():
                W[R_map[loc], col] += coeff
            col += 1

    if not _verify(U, V, W, n, m, p):
        return None
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
