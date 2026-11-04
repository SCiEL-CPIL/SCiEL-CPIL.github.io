import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _construct_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2x2
    # Each term is a tuple (u_dict, v_dict, w_dict)
    # u_dict maps (i, j) to coeff
    strassen = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    col = 0
    # Half 1: C cols 0-1, B cols 0-1 (B11) and B cols 2-3 (B21)
    # C = A1 B11 + A2 B21
    # A1 is A[0:2, 0:2], A2 is A[0:2, 2:4]
    # B11 is B[0:2, 0:2], B21 is B[2:4, 0:2]
    
    for term in strassen:
        u_dict, v_dict, w_dict = term
        # A1 B11
        for (i, j), c in u_dict.items():
            U[i * m + j][col] += c
        for (j, k), c in v_dict.items():
            V[j * p + k][col] += c
        for (k, i), c in w_dict.items():
            W[k * n + i][col] += c
        col += 1
        
        # A2 B21
        for (i, j), c in u_dict.items():
            # A2 is at cols 2,3 of A. So j_global = j + 2
            U[i * m + (j + 2)][col] += c
        for (j, k), c in v_dict.items():
            # B21 is at rows 2,3 of B. So j_global = j + 2
            V[(j + 2) * p + k][col] += c
        for (k, i), c in w_dict.items():
            W[k * n + i][col] += c
        col += 1

    # Half 2: C cols 2-3, B cols 2-3 (B12) and B cols 4-5 (B22)
    # C = A1 B12 + A2 B22
    # B12 is B[0:2, 2:4], B22 is B[2:4, 2:4]
    
    for term in strassen:
        u_dict, v_dict, w_dict = term
        # A1 B12
        for (i, j), c in u_dict.items():
            U[i * m + j][col] += c
        for (j, k), c in v_dict.items():
            # B12 is at cols 2,3 of B. So k_global = k + 2
            V[j * p + (k + 2)][col] += c
        for (k, i), c in w_dict.items():
            # C cols 2,3. So i_global? No, W index is k*n+i.
            # C is 2x4. C[0:2, 2:4].
            # The output index in W is (k, i). Here k is 0,1 (local), i is 0,1 (local).
            # Global C index is (i, k+2)?
            # W shape is (p*n, rank). W[k*n+i, r].
            # Here k is the row of C? No, W[k*n+i] corresponds to C[i,k]?
            # Prompt: W[k*n+i,r].
            # In _matmul_check: Wr is (p, n, r). Wr[k, i, r].
            # T[i,j,k] = sum Ur[i,j,r] Vr[j,k,r] Wr[k,i,r].
            # So W index is (k, i).
            # For C cols 2,3, the local k is 0,1. Global k is 2,3.
            # So k_global = k + 2.
            W[(k + 2) * n + i][col] += c
        col += 1
        
        # A2 B22
        for (i, j), c in u_dict.items():
            U[i * m + (j + 2)][col] += c
        for (j, k), c in v_dict.items():
            # B22 is at rows 2,3 and cols 2,3 of B.
            V[(j + 2) * p + (k + 2)][col] += c
        for (k, i), c in w_dict.items():
            W[(k + 2) * n + i][col] += c
        col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    cand = _construct_rank28()
    return cand
