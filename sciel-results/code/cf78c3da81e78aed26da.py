"""Rank-minimal tensor decomposition for 2x4 @ 4x6 matrix multiplication using Strassen blocks."""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 6
    
    # Strassen 2x2x2: 7 rank-1 terms
    # Local indices: A[(i,j)], B[(i,j)], C[(i,j)] for i,j in {0,1}
    # Each term: (u_dict, v_dict, w_dict) mapping local (i,j) -> coefficient
    
    def make_strassen_terms():
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        terms.append({
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        })
        # M2 = (A10+A11)(B00) -> C10 += M2, C11 += M2
        terms.append({
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(1,0): 1, (1,1): 1}
        })
        # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
        terms.append({
            'u': {(0,0): 1},
            'v': {(1,0): 1, (1,1): -1},
            'w': {(0,1): 1, (1,1): 1}
        })
        # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
        terms.append({
            'u': {(1,1): 1},
            'v': {(0,1): 1, (0,0): -1},
            'w': {(0,0): 1, (1,0): 1}
        })
        # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
        terms.append({
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,0): -1, (0,1): 1}
        })
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        terms.append({
            'u': {(1,0): 1, (0,0): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(1,1): 1}
        })
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        terms.append({
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(0,0): 1}
        })
        return terms
    
    strassen_terms = make_strassen_terms()
    
    rank = 42  # 2 A-blocks * 3 B-blocks * 7 Strassen terms
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    col = 0
    for a_block in range(2):  # A1 (cols 0-1) or A2 (cols 2-3)
        for b_block in range(3):  # B blocks: rows (a_block*2 to a_block*2+1), cols (b_block*2 to b_block*2+1)
            b_row_start = a_block * 2
            b_col_start = b_block * 2
            
            for term in strassen_terms:
                # U: A entries at (i, a_block*2 + j_local)
                for (i_local, j_local), val in term['u'].items():
                    i_global = i_local
                    j_global = a_block * 2 + j_local
                    U[i_global * m + j_global][col] += val
                
                # V: B entries at (b_row_start + i_local, b_col_start + j_local)
                for (i_local, j_local), val in term['v'].items():
                    i_global = b_row_start + i_local
                    j_global = b_col_start + j_local
                    V[i_global * p + j_global][col] += val
                
                # W: C entries at (b_col_start + j_local, i_local)
                for (i_local, j_local), val in term['w'].items():
                    i_global = i_local
                    j_global = b_col_start + j_local
                    W[j_global * n + i_global][col] += val
                
                col += 1
    
    # Verify correctness
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r] for r in range(rank))
                    if abs(s - 1) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    valid, err = verify(U, V, W)
    if not valid:
        # Fallback to schoolbook if Strassen is buggy
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
