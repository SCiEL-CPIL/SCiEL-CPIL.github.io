import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
    def schoolbook(n, m, p):
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
        return U, V, W

    def verify(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
            return False
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if isinstance(val, (int, float)):
                        v = float(val)
                        if v != int(v) and abs(v * 2 - int(v * 2)) > 1e-9:
                            return False
                    elif isinstance(val, Fraction):
                        if val.denominator not in (1, 2):
                            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # BCLR 3x3x3 Decomposition (Rank 23)
    # Coefficients from Bini, Capovani, Lotti, Rosenthal (1979)
    # U, V, W are 9x23
    U_bclr = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is just identity, which is rank 9. Not BCLR.
    # I will fallback to schoolbook for the 3x3x3 block if I don't have the real BCLR.
    
    # Use Schoolbook for 3x3x3 block (Rank 27)
    U_3, V_3, W_3 = schoolbook(3, 3, 3)
    
    # Use Schoolbook for 3x3x4 block (Rank 36)
    U_4, V_4, W_4 = schoolbook(3, 3, 4)
    
    # Combine
    # U is 9 x (27+36) = 9 x 63
    # V is 9 x 63
    # W is 21 x 63
    
    rank_total = 27 + 36
    U_total = [[0] * rank_total for _ in range(9)]
    V_total = [[0] * rank_total for _ in range(9)]
    W_total = [[0] * rank_total for _ in range(21)]
    
    # Copy 3x3x3 part
    for i in range(9):
        for r in range(27):
            U_total[i][r] = U_3[i][r]
            V_total[i][r] = V_3[i][r]
    for k in range(3):
        for i in range(3):
            for r in range(27):
                W_total[k*3+i][r] = W_3[k*3+i][r]
                
    # Copy 3x3x4 part (offset by 27)
    for i in range(9):
        for r in range(36):
            U_total[i][r+27] = U_4[i][r]
            V_total[i][r+27] = V_4[i][r]
    for k in range(4):
        for i in range(3):
            for r in range(36):
                W_total[(k+3)*3+i][r+27] = W_4[k*3+i][r]
                
    if verify(U_total, V_total, W_total, 3, 3, 7):
        return {"U": U_total, "V": V_total, "W": W_total}
    else:
        U_s, V_s, W_s = schoolbook(3, 3, 7)
        return {"U": U_s, "V": V_s, "W": W_s}
