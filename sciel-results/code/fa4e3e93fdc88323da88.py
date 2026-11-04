import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    school_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Improved decomposition using Strassen for 2x2x2 blocks
    # Total rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(4):
        # Block k corresponds to B columns 2k, 2k+1
        # A indices: 00, 01, 02, 10, 11, 12 (row*3+col)
        # B indices: 00..07, 10..17, 20..27 (row*8+col)
        # W indices: 00..07, 10..17 (col*2+row) -> k*n+i
        
        # 4 direct terms for A_col2 * B_row2
        # t1: A_02 * B_2(2k) -> C_0(2k)
        U[0*3 + 2][col] = 1
        V[2*8 + 2*k][col] = 1
        W[2*k*2 + 0][col] = 1
        col += 1
        
        # t2: A_02 * B_2(2k+1) -> C_0(2k+1)
        U[0*3 + 2][col] = 1
        V[2*8 + 2*k + 1][col] = 1
        W[(2*k+1)*2 + 0][col] = 1
        col += 1
        
        # t3: A_12 * B_2(2k) -> C_1(2k)
        U[1*3 + 2][col] = 1
        V[2*8 + 2*k][col] = 1
        W[2*k*2 + 1][col] = 1
        col += 1
        
        # t4: A_12 * B_2(2k+1) -> C_1(2k+1)
        U[1*3 + 2][col] = 1
        V[2*8 + 2*k + 1][col] = 1
        W[(2*k+1)*2 + 1][col] = 1
        col += 1
        
        # 7 Strassen terms for A_01, A_11 * B_01, B_11
        # A: a=A_00, b=A_01, c=A_10, d=A_11
        # B: e=B_0(2k), f=B_0(2k+1), g=B_1(2k), h=B_1(2k+1)
        
        # M1: (a+d)(e+h)
        # U: a, d
        # V: e, h
        # W: C_0(2k), C_0(2k+1), C_1(2k), C_1(2k+1) all +1
        U[0*3 + 0][col] = 1
        U[1*3 + 1][col] = 1
        V[0*8 + 2*k][col] = 1
        V[1*8 + 2*k + 1][col] = 1
        W[2*k*2 + 0][col] = 1
        W[(2*k+1)*2 + 0][col] = 1
        W[2*k*2 + 1][col] = 1
        W[(2*k+1)*2 + 1][col] = 1
        col += 1
        
        # M2: (c+d)e
        # U: c, d
        # V: e
        # W: C_1(2k), C_1(2k+1) all +1
        U[1*3 + 0][col] = 1
        U[1*3 + 1][col] = 1
        V[0*8 + 2*k][col] = 1
        W[2*k*2 + 1][col] = 1
        W[(2*k+1)*2 + 1][col] = 1
        col += 1
        
        # M3: a(f-h)
        # U: a
        # V: f, -h
        # W: C_0(2k+1), C_1(2k+1) all +1
        U[0*3 + 0][col] = 1
        V[0*8 + 2*k + 1][col] = 1
        V[1*8 + 2*k + 1][col] = -1
        W[(2*k+1)*2 + 0][col] = 1
        W[(2*k+1)*2 + 1][col] = 1
        col += 1
        
        # M4: d(g-e)
        # U: d
        # V: g, -e
        # W: C_0(2k), C_1(2k) all +1
        U[1*3 + 1][col] = 1
        V[1*8 + 2*k][col] = 1
        V[0*8 + 2*k][col] = -1
        W[2*k*2 + 0][col] = 1
        W[2*k*2 + 1][col] = 1
        col += 1
        
        # M5: (a+b)h
        # U: a, b
        # V: h
        # W: C_0(2k) -1, C_0(2k+1) +1
        U[0*3 + 0][col] = 1
        U[0*3 + 1][col] = 1
        V[1*8 + 2*k + 1][col] = 1
        W[2*k*2 + 0][col] = -1
        W[(2*k+1)*2 + 0][col] = 1
        col += 1
        
        # M6: (c-a)(e+f)
        # U: c, -a
        # V: e, f
        # W: C_1(2k+1) +1
        U[1*3 + 0][col] = 1
        U[0*3 + 0][col] = -1
        V[0*8 + 2*k][col] = 1
        V[0*8 + 2*k + 1][col] = 1
        W[(2*k+1)*2 + 1][col] = 1
        col += 1
        
        # M7: (b-d)(g+h)
        # U: b, -d
        # V: g, h
        # W: C_0(2k) +1, C_1(2k+1) -1
        U[0*3 + 1][col] = 1
        U[1*3 + 1][col] = -1
        V[1*8 + 2*k][col] = 1
        V[1*8 + 2*k + 1][col] = 1
        W[2*k*2 + 0][col] = 1
        W[(2*k+1)*2 + 1][col] = -1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
