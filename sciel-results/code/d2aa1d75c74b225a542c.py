"""Matrix multiplication 2x4 x 4x6 — rank-minimal exact tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        uc = np.array([U[i][c] for i in range(n * m)], dtype=float)
        vc = np.array([V[i][c] for i in range(m * p)], dtype=float)
        wc = np.array([W[i][c] for i in range(p * n)], dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    target = np.zeros_like(T)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, target, atol=1e-9)

def _schoolbook(n, m, p):
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

def _strassen_block_decomposition(n, m, p):
    """
    Decompose M_{n,m,p} using block Strassen.
    For (2,4,6): split m=4 into two blocks of size 2, p=6 into three blocks of size 2.
    Each subproblem is M_{2,2,2} solved by Strassen (rank 7).
    Total rank = 2 * 3 * 7 = 42.
    """
    a_block_size = 2
    b_col_block_size = 2
    a_blocks = m // a_block_size  # 2
    b_col_blocks = p // b_col_block_size  # 3
    strassen_rank = 7
    rank = a_blocks * b_col_blocks * strassen_rank  # 42

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen for 2x2 * 2x2 -> 2x2
    # M = [[a,b],[c,d]], N = [[e,f],[g,h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # R = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]

    # U coefficients: (i_local, j_local) -> coeff for each Strassen term
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # P1: a + d
        {(1, 0): 1, (1, 1): 1},   # P2: c + d
        {(0, 0): 1},               # P3: a
        {(1, 1): 1},               # P4: d
        {(0, 0): 1, (0, 1): 1},   # P5: a + b
        {(1, 0): 1, (0, 0): -1},  # P6: c - a
        {(0, 1): 1, (1, 1): -1}   # P7: b - d
    ]

    # V coefficients: (j_local, k_local) -> coeff
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # P1: e + h
        {(0, 0): 1},               # P2: e
        {(0, 1): 1, (1, 1): -1},  # P3: f - h
        {(1, 0): 1, (0, 0): -1},  # P4: g - e
        {(1, 1): 1},               # P5: h
        {(0, 0): 1, (0, 1): 1},   # P6: e + f
        {(1, 0): 1, (1, 1): 1}    # P7: g + h
    ]

    # W coefficients: (i_local, k_local) -> coeff
    # R00 = P1 + P4 - P5 + P7
    # R01 = P3 + P5
    # R10 = P2 + P4
    # R11 = P1 - P2 + P3 + P6
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # P1
        {(1, 0): 1, (1, 1): -1},  # P2
        {(0, 1): 1, (1, 1): 1},   # P3
        {(0, 0): 1, (1, 0): 1},   # P4
        {(0, 0): -1, (0, 1): 1},  # P5
        {(1, 1): 1},              # P6
        {(0, 0): 1}               # P7
    ]

    col = 0
    for a in range(a_blocks):
        for b in range(b_col_blocks):
            for s in range(strassen_rank):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * a_block_size + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff

                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * a_block_size + j_loc
                    k = b * b_col_block_size + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff

                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * b_col_block_size + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff

                col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110

    # Try block Strassen decomposition (rank 42)
    try:
        result = _strassen_block_decomposition(n, m, p)
        if _verify(n, m, p, result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass

    # Fallback to schoolbook (rank 48)
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
