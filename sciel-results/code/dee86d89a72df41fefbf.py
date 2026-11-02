"""Tensor decomposition for 2x4x4 matrix multiplication (rank-minimal search)."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

# Strassen 2x2x2 decomposition: 7 products.
# M1 = A11*(B12 - B22)
# M2 = (A11 + A21)*B22
# M3 = (A12 - A11)*B11
# M4 = A22*(B11 - B21)
# M5 = (A11 + A12)*B12
# M6 = (A21 - A22)*(B11 + B22)
# M7 = (A12 - A21)*(B21 + B22)
# C11 = M5 + M4 - M2 + M6
# C12 = M1 + M5
# C21 = M2 + M3
# C22 = M1 - M3 + M4 + M7
_STRA = {
    0: [([0, 0, 1, 0], [0, 1, 0, 0]),
        ([-1, 0, 0, 1], [0, 0, 0, 1])],
    1: [([1, 0, 0, 1], [0, 0, 0, 1]),
        ([0, 0, 0, 0], [0, 0, 1, 1])],
    2: [([1, -1, 0, 0], [0, 0, 1, 0]),
        ([0, 0, 0, 0], [1, 1, 0, 0])],
    3: [([0, 0, 0, 1], [1, 0, 0, -1]),
        ([0, 0, 0, 0], [1, 0, 0, 0])],
    4: [([1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 0, 0, 0], [0, 1, 1, 0])],
    5: [([0, 0, 1, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 0], [1, 1, 1, 1])],
    6: [([0, 0, 1, -1], [1, 0, 0, 0]),
        ([0, 0, 0, 0], [0, 0, 1, 1])],
}
_STRA_W = {
    0: [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    1: [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    2: [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
    3: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
    4: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    5: [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    6: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
}

def _strassen_block(u_base, v_base, w_base, a_row, b_col, c_row):
    """Emit 7 Strassen products for one 2x2x2 block.
    a_row: U row index of A11.  b_col: V row index of B11.  c_row: W row index of C11.
    """
    for t in range(7):
        u_f, v_f = _STRA[t]
        for a in range(4):
            u_base[a_row + a][t] += u_f[a]
        for b in range(4):
            v_base[b_col + b][t] += v_f[b]
        for c in range(4):
            w_base[c_row + c][t] += _STRA_W[t][c]

def _build_rank14():
    """Two Strassen blocks: (A11,B11)->C11 and (A22,B22)->C22. rank 14."""
    n, m, p = 2, 4, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]   # 8 x 14
    V = [[0] * rank for _ in range(m * p)]   # 16 x 14
    W = [[0] * rank for _ in range(p * n)]   # 8 x 14
    _strassen_block(U, V, W, a_row=0, b_col=0, c_row=0)   # C11 (rows 0-1)
    _strassen_block(U, V, W, a_row=2, b_col=8, c_row=2)   # C22 (rows 2-3)
    return {"U": U, "V": V, "W": W}

def _verify(result, tol=1e-9):
    """Check sum_c U[i,c]*V[j,c]*W[k*n+i,c] == delta(i+j,k)."""
    n, m, p = 2, 4, 4
    U = result["U"]; V = result["V"]; W = result["W"]
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                target = 1.0 if (i + j == k) else 0.0
                if abs(s - target) > tol:
                    return False
    return True

def _verify_schoolbook():
    n, m, p = 2, 4, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 120.0
    fallback = _verify_schoolbook()

    # Primary candidate: rank-14 Strassen block decomposition.
    cand = _build_rank14()
    if _verify(cand):
        return cand
    # If verification unexpectedly fails, fall back to schoolbook (rank 32).
    return fallback

# EVOLVE-BLOCK-END
