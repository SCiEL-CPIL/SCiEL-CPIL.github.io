"""Program for rank-minimal tensor decomposition of 4x5 * 5x5 matrix multiplication."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110  # safety margin before 120s limit
    
    i_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    
    terms = []  # list of (u_entries, v_entries, w_entries) where each is list of (idx, coeff)
    
    def add_bilinear_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    for ib in i_blocks:
        for kb in k_blocks:
            for jb in j_blocks:
                i_size = len(ib)
                k_size = len(kb)
                j_size = len(jb)
                
                if i_size == 2 and k_size == 2 and j_size == 2:
                    # Strassen: 7 bilinear terms
                    # Each term: (list of (u_idx, coeff), list of (v_idx, coeff), list of (w_idx, coeff))
                    
                    # M1 = (A11+A22)(B11+B22)
                    u_M1 = [(ib[0]*m+jb[0], 1), (ib[1]*m+jb[1], 1)]
                    v_M1 = [(jb[0]*p+kb[0], 1), (jb[1]*p+kb[1], 1)]
                    # Contributes to C[ib[0],kb[0]] with +1, C[ib[1],kb[1]] with +1
                    w_M1 = [(kb[0]*n+ib[0], 1), (kb[1]*n+ib[1], 1)]
                    add_bilinear_term(u_M1, v_M1, w_M1)
                    
                    # M2 = (A21+A22)B11
                    u_M2 = [(ib[1]*m+jb[0], 1), (ib[1]*m+jb[1], 1)]
                    v_M2 = [(jb[0]*p+kb[0], 1)]
                    # Contributes to C[ib[1],kb[0]] with +1, C[ib[1],kb[1]] with -1
                    w_M2 = [(kb[0]*n+ib[1], 1), (kb[1]*n+ib[1], -1)]
                    add_bilinear_term(u_M2, v_M2, w_M2)
                    
                    # M3 = A11(B12-B22)
                    u_M3 = [(ib[0]*m+jb[0], 1)]
                    v_M3 = [(jb[0]*p+kb[1], 1), (jb[1]*p+kb[1], -1)]
                    # Contributes to C[ib[0],kb[1]] with +1, C[ib[1],kb[1]] with +1
                    w_M3 = [(kb[1]*n+ib[0], 1), (kb[1]*n+ib[1], 1)]
                    add_bilinear_term(u_M3, v_M3, w_M3)
                    
                    # M4 = A22(B21-B11)
                    u_M4 = [(ib[1]*m+jb[1], 1)]
                    v_M4 = [(jb[1]*p+kb[0], 1), (jb[0]*p+kb[0], -1)]
                    # Contributes to C[ib[0],kb[0]] with -1, C[ib[1],kb[0]] with +1
                    w_M4 = [(kb[0]*n+ib[0], -1), (kb[0]*n+ib[1], 1)]
                    add_bilinear_term(u_M4, v_M4, w_M4)
                    
                    # M5 = (A11+A12)B22
                    u_M5 = [(ib[0]*m+jb[0], 1), (ib[0]*m+jb[1], 1)]
                    v_M5 = [(jb[1]*p+kb[1], 1)]
                    # Contributes to C[ib[0],kb[0]] with +1, C[ib[0],kb[1]] with +1
                    w_M5 = [(kb[0]*n+ib[0], 1), (kb[1]*n+ib[0], 1)]
                    add_bilinear_term(u_M5, v_M5, w_M5)
                    
                    # M6 = (A21-A11)(B11+B12)
                    u_M6 = [(ib[1]*m+jb[0], 1), (ib[0]*m+jb[0], -1)]
                    v_M6 = [(jb[0]*p+kb[0], 1), (jb[0]*p+kb[1], 1)]
                    # Contributes to C[ib[1],kb[1]] with +1
                    w_M6 = [(kb[1]*n+ib[1], 1)]
                    add_bilinear_term(u_M6, v_M6, w_M6)
                    
                    # M7 = (A12-A22)(B21+B22)
                    u_M7 = [(ib[0]*m+jb[1], 1), (ib[1]*m+jb[1], -1)]
                    v_M7 = [(jb[1]*p+kb[0], 1), (jb[1]*p+kb[1], 1)]
                    # Contributes to C[ib[0],kb[0]] with -1
                    w_M7 = [(kb[0]*n+ib[0], -1)]
                    add_bilinear_term(u_M7, v_M7, w_M7)
                    
                else:
                    # Schoolbook: each term is a single (u_idx, v_idx, w_idx)
                    for i in ib:
                        for k in kb:
                            for j in jb:
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                add_bilinear_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
    
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for (idx, coeff) in u_entries:
            U[idx, col] += coeff
        for (idx, coeff) in v_entries:
            V[idx, col] += coeff
        for (idx, coeff) in w_entries:
            W[idx, col] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
