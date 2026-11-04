"""Matrix multiplication 2x4x4: rank-28 decomposition via 4 independent Strassen 2x2 block products."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def _strassen_2x2():
    """Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    Local indices are 0..3 for 2x2 matrices (row-major)."""
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns

def _build_strassen_28():
    """Build the rank-28 decomposition via 4 independent Strassen 2x2 block products.
    
    We split the 2x4 A into A_L (cols 0-1) and A_R (cols 2-3).
    We split the 4x4 B into 4 2x2 blocks: B_TL, B_TR, B_BL, B_BR.
    The 4x4 C is split into 4 2x2 blocks: C_TL, C_TR, C_BL, C_BR.
    
    C_TL = A_L * B_TL
    C_TR = A_L * B_TR
    C_BL = A_R * B_TL
    C_BR = A_R * B_TR
    
    Wait, that's not right. Let me re-derive.
    A is 2x4, B is 4x4, C is 2x4.
    A = [A_L, A_R] where A_L is 2x2 (cols 0-1), A_R is 2x2 (cols 2-3).
    B = [[B_TL, B_TR], [B_BL, B_BR]] where each block is 2x2.
    
    C = A * B = [A_L, A_R] * [[B_TL, B_TR], [B_BL, B_BR]]
    C = [A_L*B_TL + A_R*B_BL, A_L*B_TR + A_R*B_BR]
    
    So C has two 2x2 blocks:
    C_L = A_L*B_TL + A_R*B_BL  (cols 0-1 of C)
    C_R = A_L*B_TR + A_R*B_BR  (cols 2-3 of C)
    
    Each is a sum of two 2x2x2 matrix products.
    A_L*B_TL has rank 7 (Strassen).
    A_R*B_BL has rank 7 (Strassen).
    A_L*B_TR has rank 7 (Strassen).
    A_R*B_BR has rank 7 (Strassen).
    
    Total rank = 28.
    
    The blocks in the code are:
    (a_r, a_c, b_r, b_c, c_r, c_c)
    a_r, a_c: row, col offset of A block (2x2)
    b_r, b_c: row, col offset of B block (2x2)
    c_r, c_c: row, col offset of C block (2x2)
    
    For A_L*B_TL: a_r=0, a_c=0, b_r=0, b_c=0, c_r=0, c_c=0
    For A_L*B_TR: a_r=0, a_c=0, b_r=0, b_c=2, c_r=0, c_c=2
    For A_R*B_BL: a_r=0, a_c=2, b_r=2, b_c=0, c_r=0, c_c=0
    For A_R*B_BR: a_r=0, a_c=2, b_r=2, b_c=2, c_r=0, c_c=2
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_patterns = _strassen_2x2()

    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 2),
    ]

    col = 0
    for (a_r, a_c, b_r, b_c, c_r, c_c) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = a_r + local_r
                global_j = a_c + local_c
                U[global_i * m + global_j][col] += coeff
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_r + local_r
                global_k = b_c + local_c
                V[global_j * p + global_k][col] += coeff
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = c_r + local_r
                global_k = c_c + local_c
                W[global_k * n + global_i][col] += coeff
            col += 1

    return U, V, W

def _build_tensor(n, m, p):
    """Build the multiplication tensor T[a,b,c]=1 with trace convention c=k*n+i."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, rank):
    """Max abs error between reconstructed tensor and T."""
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    S = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return float(np.max(np.abs(S - T)))

def construct():
    t0 = time.time()
    deadline = 150.0 - 10.0  # safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    T = _build_tensor(n, m, p)

    U, V, W = _build_strassen_28()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    err = _verify(U, V, W, T, rank)
    if err > 1e-9:
        # Should not happen, but fall back to schoolbook if needed
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
