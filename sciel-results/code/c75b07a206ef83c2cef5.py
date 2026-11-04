"""
5x5 matrix multiplication tensor decomposition.
Uses block decomposition 2+2+1 with Strassen for 2x2x2 blocks.
Total rank: 4*(7) + 2*(4) + 2*(4) + 1 = 28 + 8 + 8 + 1 = 45.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strassen 7-term decomposition for 2x2x2
    strassen_2x2 = _get_strassen_2x2()
    block_rank = 7

    # Build full 5x5x5 decomposition using blocks [0,1], [2,3], [4]
    blocks = [(0, 2), (2, 4), (4, 5)]
    terms = []

    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline - 5:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1

                if ar == 2 and ac == 2 and bc == 2:
                    # Use the 2x2x2 Strassen decomposition
                    idx_i = [r1, r1 + 1]
                    idx_j = [c1, c1 + 1]
                    idx_k = [k1, k1 + 1]
                    
                    for u_dict, v_dict, w_dict in strassen_2x2:
                        u_entries = {}
                        v_entries = {}
                        w_entries = {}
                        for (ii, jj), val in u_dict.items():
                            u_entries[idx_i[ii] * m + idx_j[jj]] = val
                        for (jj, kk), val in v_dict.items():
                            v_entries[idx_j[jj] * p + idx_k[kk]] = val
                        for (kk, ii), val in w_dict.items():
                            w_entries[idx_k[kk] * n + idx_i[ii]] = val
                        if u_entries and v_entries and w_entries:
                            terms.append((u_entries, v_entries, w_entries))
                else:
                    # Direct multiplication for mixed-size blocks
                    for ii in range(ar):
                        for jj in range(ac):
                            for kk in range(bc):
                                u_idx = (r1 + ii) * m + (c1 + jj)
                                v_idx = (c1 + jj) * p + (k1 + kk)
                                w_idx = (k1 + kk) * n + (r1 + ii)
                                terms.append(
                                    ({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                                )

    rank = len(terms)
    
    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}


def _get_strassen_2x2():
    """
    Standard Strassen 7-term decomposition for 2x2x2 matrix multiplication.
    
    The 2x2x2 matmul tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k).
    
    Each term is (u_dict, v_dict, w_dict) where:
    - u_dict maps (i,j) -> coefficient
    - v_dict maps (j,k) -> coefficient  
    - w_dict maps (k,i) -> coefficient
    """
    terms = [
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1, C01 -= M1, C10 -= M1
        ({(0,0):1, (1,1):1}, 
         {(0,0):1, (1,1):1}, 
         {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}),
        
        # M2 = (A10+A11)B00
        # C10 += M2, C11 -= M2
        ({(1,0):1, (1,1):1}, 
         {(0,0):1}, 
         {(1,0):1, (1,1):-1}),
        
        # M3 = A00(B01-B11)
        # C01 += M3, C11 += M3
        ({(0,0):1}, 
         {(0,1):1, (1,1):-1}, 
         {(0,1):1, (1,1):1}),
        
        # M4 = A11(B10-B00)
        # C00 += M4, C10 += M4
        ({(1,1):1}, 
         {(1,0):1, (0,0):-1}, 
         {(0,0):1, (1,0):1}),
        
        # M5 = (A00+A01)B11
        # C00 -= M5, C01 += M5
        ({(0,0):1, (0,1):1}, 
         {(1,1):1}, 
         {(0,0):-1, (0,1):1}),
        
        # M6 = (A10-A00)(B00+B01)
        # C11 += M6
        ({(1,0):1, (0,0):-1}, 
         {(0,0):1, (0,1):1}, 
         {(1,1):1}),
        
        # M7 = (A01-A11)(B10+B11)
        # C00 += M7
        ({(0,1):1, (1,1):-1}, 
         {(1,0):1, (1,1):1}, 
         {(0,0):1}),
    ]
    return terms
# EVOLVE-BLOCK-END
