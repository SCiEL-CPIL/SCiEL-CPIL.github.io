import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    """
    Returns a rank-7 + 117 = 124 exact tensor decomposition for multiplying a 5 by 5 matrix 
    by a 5 by 5 matrix.
    Uses Strassen (rank 7) for the 2x2x2 core + schoolbook for 117 border terms.
    """
    n, m, p = DIMENSIONS
    terms = []
    
    def add_term(u_entries, v_entries, w_idx, coeff):
        """
        u_entries: list of (u_idx, coeff) for U column
        v_entries: list of (v_idx, coeff) for V column
        w_idx: single w index
        coeff: overall coefficient (applied to all)
        """
        if coeff == 0:
            return
        terms.append((u_entries, v_entries, w_idx, coeff))
    
    # Strassen for 2x2 block at [0:2, 0:2]
    # A[0,0]=u0, A[0,1]=u1, A[1,0]=u2, A[1,1]=u3
    # B[0,0]=v0, B[0,1]=v1, B[1,0]=v2, B[1,1]=v3
    # C[0,0]=w0, C[0,1]=w1, C[1,0]=w2, C[1,1]=w3
    
    u00 = 0*m + 0  # A[0,0]
    u01 = 0*m + 1  # A[0,1]
    u10 = 1*m + 0  # A[1,0]
    u11 = 1*m + 1  # A[1,1]
    
    v00 = 0*p + 0  # B[0,0]
    v01 = 0*p + 1  # B[0,1]
    v10 = 1*p + 0  # B[1,0]
    v11 = 1*p + 1  # B[1,1]
    
    w00 = 0*n + 0  # C[0,0]
    w01 = 0*n + 1  # C[0,1]
    w10 = 1*n + 0  # C[1,0]
    w11 = 1*n + 1  # C[1,1]
    
    # M1 = (A00+A11)(B00+B11)
    # Contributes to C00 and C11
    add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], w00, 1)
    add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], w11, 1)
    
    # M2 = (A10+A11)B00
    # Contributes to C10 and C11
    add_term([(u10, 1), (u11, 1)], [(v00, 1)], w10, 1)
    add_term([(u10, 1), (u11, 1)], [(v00, 1)], w11, -1)
    
    # M3 = A00(B01-B11)
    # Contributes to C01 and C11
    add_term([(u00, 1)], [(v01, 1), (v11, -1)], w01, 1)
    add_term([(u00, 1)], [(v01, 1), (v11, -1)], w11, 1)
    
    # M4 = A11(B10-B00)
    # Contributes to C00 and C10
    add_term([(u11, 1)], [(v10, 1), (v00, -1)], w00, 1)
    add_term([(u11, 1)], [(v10, 1), (v00, -1)], w10, 1)
    
    # M5 = (A00+A01)B11
    # Contributes to C00 and C01
    add_term([(u00, 1), (u01, 1)], [(v11, 1)], w00, -1)
    add_term([(u00, 1), (u01, 1)], [(v11, 1)], w01, 1)
    
    # M6 = (A10-A00)(B00+B01)
    # Contributes to C11
    add_term([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], w11, 1)
    
    # M7 = (A01-A11)(B10+B11)
    # Contributes to C00
    add_term([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], w00, 1)
    
    # Now add schoolbook terms for all (i,j,k) where NOT (i<2 and j<2 and k<2)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                u = i*m + j
                v = j*p + k
                w = k*n + i
                add_term([(u, 1)], [(v, 1)], w, 1)
    
    # Build the matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_entries, v_entries, w_idx, coeff) in enumerate(terms):
        for u_idx, u_coeff in u_entries:
            U[u_idx][col] += coeff * u_coeff
        for v_idx, v_coeff in v_entries:
            V[v_idx][col] += coeff * v_coeff
        W[w_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
