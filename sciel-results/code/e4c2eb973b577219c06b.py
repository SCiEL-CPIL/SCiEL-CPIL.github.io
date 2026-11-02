import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Split: n=[0]+[1,2], m=[0,1]+[2,3], p=[0,1]+[2,3]+[4,5]
    # Blocks: (1,2,2) x1, (2,2,2) x6
    # Rank: 4 + 6*7 = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 12 x 46
    V = [[0] * rank for _ in range(m * p)]  # 24 x 46
    W = [[0] * rank for _ in range(p * n)]  # 18 x 46
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
    
    # Block (1,2,2): i in [0], j in [0,1], k in [0,1]
    add_term(0*2+0, 0*2+0, 0*3+0)  # A[0,0]*B[0,0] -> C[0,0]
    add_term(0*2+1, 1*2+0, 0*3+0)  # A[0,1]*B[1,0] -> C[0,0]
    add_term(0*2+0, 0*2+1, 1*3+0)  # A[0,0]*B[0,1] -> C[0,1]
    add_term(0*2+1, 1*2+1, 1*3+0)  # A[0,1]*B[1,1] -> C[0,1]
    
    # Strassen for 2x2x2
    strassen_u = [
        [1,0,0,1],  # M1: A00+A11
        [0,0,1,1],  # M2: A10+A11
        [1,0,0,0],  # M3: A00
        [0,0,0,1],  # M4: A11
        [1,1,0,0],  # M5: A00+A01
        [0,0,1,-1], # M6: A10-A00
        [0,1,0,-1], # M7: A01-A11
    ]
    strassen_v = [
        [1,0,0,1],  # M1: B00+B11
        [1,0,0,0],  # M2: B00
        [0,1,0,-1], # M3: B01-B11
        [0,0,1,-1], # M4: B10-B00
        [0,0,0,1],  # M5: B11
        [1,1,0,0],  # M6: B00+B01
        [0,0,1,1],  # M7: B10+B11
    ]
    strassen_w = [
        [1,0,0,1],  # M1: C00+C11
        [0,0,1,0],  # M2: C10
        [0,1,0,0],  # M3: C01
        [0,0,1,0],  # M4: C10
        [0,1,0,1],  # M5: C01+C11
        [1,0,1,0],  # M6: C00+C10
        [1,0,0,1],  # M7: C00+C11
    ]
    
    def strassen_block(i_base, j_base, k_base):
        nonlocal col
        def a_idx(ia, ja):
            return (i_base + ia) * m + (j_base + ja)
        def b_idx(jb, kb):
            return (j_base + jb) * p + (k_base + kb)
        def c_idx(ic, kc):
            return (k_base + kc) * n + (i_base + ic)
        
        for t in range(7):
            c = col
            col += 1
            for ia in range(2):
                for ja in range(2):
                    val = strassen_u[t][ia*2+ja]
                    if val != 0:
                        U[a_idx(ia, ja)][c] += val
            for jb in range(2):
                for kb in range(2):
                    val = strassen_v[t][jb*2+kb]
                    if val != 0:
                        V[b_idx(jb, kb)][c] += val
            for ic in range(2):
                for kc in range(2):
                    val = strassen_w[t][ic*2+kc]
                    if val != 0:
                        W[c_idx(ic, kc)][c] += val
    
    # 6 (2,2,2) blocks
    for i_base in [1, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2, 4]:
                strassen_block(i_base, j_base, k_base)
    
    return {"U": U, "V": V, "W": W}
