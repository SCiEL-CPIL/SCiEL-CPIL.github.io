"""Rank-46 Strassen-based decomposition for 2x5 x 5x5 matrix multiplication."""

import time

DEADLINE = time.time() + 130


def _schoolbook(n=2, m=5, p=5):
    """Standard schoolbook rank-50 decomposition."""
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


def _strassen_terms():
    """Return 7 Strassen terms for 2x2 * 2x2 -> 2x2.
    
    Each term is (U_dict, V_dict, W_dict) where:
    - U_dict maps (i, j) -> coeff for A[i,j]
    - V_dict maps (j, k) -> coeff for B[j,k]
    - W_dict maps (k, i) -> coeff for C[i,k]
    
    Strassen:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    # C00 += M1 (k=0,i=0), C11 += M1 (k=1,i=1)
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # M2 = (A10+A11)*B00
    # C10 += M2 (k=0,i=1), C11 -= M2 (k=1,i=1)
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    # M3 = A00*(B01-B11)
    # C01 += M3 (k=1,i=0), C11 += M3 (k=1,i=1)
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    # M4 = A11*(B10-B00)
    # C00 += M4 (k=0,i=0), C10 += M4 (k=0,i=1)
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    # M5 = (A00+A01)*B11
    # C00 -= M5 (k=0,i=0), C01 += M5 (k=1,i=0)
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    # M6 = (A10-A00)*(B00+B01)
    # C11 += M6 (k=1,i=1)
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    # M7 = (A01-A11)*(B10+B11)
    # C00 += M7 (k=0,i=0)
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    
    return terms


def _construct_rank46():
    """Construct rank-46 decomposition using Strassen on 2+2+1 partition."""
    n, m, p = 2, 5, 5
    U = [[0] * 46 for _ in range(n * m)]
    V = [[0] * 46 for _ in range(m * p)]
    W = [[0] * 46 for _ in range(p * n)]
    col = 0
    
    strassen = _strassen_terms()
    
    # Define blocks
    J_blocks = [[0, 1], [2, 3], [4]]  # A column blocks
    R_blocks = [[0, 1], [2, 3], [4]]  # B row blocks
    C_blocks = [[0, 1], [2, 3], [4]]  # B column blocks
    
    # For each combination of J_block, R_block, C_block
    for jb_idx, J in enumerate(J_blocks):
        for rb_idx, R in enumerate(R_blocks):
            for cb_idx, C in enumerate(C_blocks):
                if len(J) == 2 and len(R) == 2 and len(C) == 2:
                    # Use Strassen: 7 terms
                    for u_dict, v_dict, w_dict in strassen:
                        for (i, j), coeff_u in u_dict.items():
                            U[i * m + J[j]][col] += coeff_u
                        for (j, k), coeff_v in v_dict.items():
                            V[R[j] * p + C[k]][col] += coeff_v
                        for (k, i), coeff_w in w_dict.items():
                            W[C[k] * n + i][col] += coeff_w
                        col += 1
                else:
                    # Use schoolbook
                    for i in range(n):
                        for j in J:
                            for k in C:
                                # Need B[j,k] to be nonzero, which requires j in R
                                if j in R:
                                    U[i * m + j][col] = 1
                                    V[j * p + k][col] = 1
                                    W[k * n + i][col] = 1
                                    col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Return a tensor decomposition for 2x5 x 5x5 matrix multiplication."""
    start = time.time()
    
    # Try rank-46 first
    result = _construct_rank46()
    
    return result
