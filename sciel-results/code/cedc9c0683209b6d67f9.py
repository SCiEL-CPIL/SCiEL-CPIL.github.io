import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    rank = 50
    U = [[0]*rank for _ in range(n*m)]   # 8 x 50
    V = [[0]*rank for _ in range(m*p)]   # 28 x 50
    W = [[0]*rank for _ in range(p*n)]   # 14 x 50
    col = 0

    def add_strassen(block, B_col_start):
        """Add 7 Strassen terms for 2x2x2: A[2x2] (block cols) x B[2x2] (block rows, B_col_start:B_col_start+2) -> C[2x2]"""
        nonlocal col
        # A[i, block*2+jl] -> U[i*4 + block*2 + jl]
        # B[block*2+jl, B_col_start+kl] -> V[(block*2+jl)*7 + B_col_start+kl]
        # C[i, B_col_start+kl] -> W[(B_col_start+kl)*2 + i]
        
        AU = lambda i, jl: i*4 + block*2 + jl
        BV = lambda jl, kl: (block*2+jl)*7 + B_col_start+kl
        CW = lambda kl, i: (B_col_start+kl)*2 + i
        
        terms = [
            # p1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # p2: (A10+A11)B00 -> C10(+1), C11(-1)
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            # p3: A00(B01-B11) -> C01(+1), C11(+1)
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            # p4: A11(B10-B00) -> C00(+1), C10(+1)
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            # p5: (A00+A01)B11 -> C00(-1), C01(+1)
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
            # p6: (A10-A00)(B00+B10) -> C11(+1)
            ([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,1,1)]),
            # p7: (A01-A11)(B10+B11) -> C00(+1)
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        for u_e, v_e, w_e in terms:
            for i, jl, v in u_e:
                U[AU(i, jl)][col] += v
            for jl, kl, v in v_e:
                V[BV(jl, kl)][col] += v
            for kl, i, v in w_e:
                W[CW(kl, i)][col] += v
            col += 1

    def schoolbook_2x2x1(block, B_col):
        """Add 4 schoolbook terms for 2x2x1: A[2x2] x B[2x1] -> C[2x1]"""
        nonlocal col
        for i in range(2):
            for jl in range(2):
                U[i*4 + block*2 + jl][col] = 1
                V[(block*2+jl)*7 + B_col][col] = 1
                W[B_col*2 + i][col] = 1
                col += 1

    for block in range(2):
        for g in range(3):
            add_strassen(block, g*2)
        schoolbook_2x2x1(block, 6)

    assert col == 50, f"Rank mismatch: {col}"
    return {"U": U, "V": V, "W": W}
