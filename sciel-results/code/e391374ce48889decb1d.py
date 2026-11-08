import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Returns a rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    The decomposition uses a 3x3 block partition where each 2x2 block multiplication
    uses Strassen's 7-term algorithm.
    Rank = 3^3 * 7 = 189.
    
    The factor matrices U (shape 36x189), V (shape 36x189), W (shape 36x189)
    satisfy the tensor decomposition identity.
    W follows the trace convention W[k*n+i, r].
    """
    deadline = time.time() + 120.0
    n, m, p = DIMENSIONS  # 6, 6, 6

    # Strassen's 7-term algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    #
    # M1 = (a+d)(e+h)  -> C11 += M1, C22 += M1
    # M2 = (c+d)e      -> C21 += M2, C22 -= M2
    # M3 = a(f-h)      -> C12 += M3, C22 += M3
    # M4 = d(g-e)      -> C11 += M4, C21 += M4
    # M5 = (a+b)h      -> C11 -= M5, C12 += M5
    # M6 = (c-a)(e+f)  -> C22 += M6
    # M7 = (b-d)(g+h)  -> C11 += M7
    #
    # Patterns: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: (row, col) -> val for A block
    # v_coeffs: (row, col) -> val for B block
    # w_coeffs: (row, col) -> val for C block (local indices)
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    rank = 27 * 7  # 189
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))

    col_idx = 0
    for I in range(3):       # block row of A / block row of C
        for J in range(3):   # block col of A / block row of B
            for K in range(3):  # block col of B / block col of C
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # U factor: maps to A's block (I, J)
                    # U row index: (2*I + pi) * 6 + (2*J + pj)
                    for (pi, pj), val in u_coeffs.items():
                        r = 2 * I + pi
                        c = 2 * J + pj
                        U[r * 6 + c, col_idx] += val
                    
                    # V factor: maps to B's block (J, K)
                    # V row index: (2*J + pi) * 6 + (2*K + pj)
                    for (pi, pj), val in v_coeffs.items():
                        r = 2 * J + pi
                        c = 2 * K + pj
                        V[r * 6 + c, col_idx] += val
                    
                    # W factor: maps to C's block (I, K)
                    # W uses trace convention: W[k*n + i, r]
                    # i = row in C, k = col in C
                    for (pi, pj), val in w_coeffs.items():
                        i = 2 * I + pi
                        k = 2 * K + pj
                        W[k * 6 + i, col_idx] += val
                    
                    col_idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
