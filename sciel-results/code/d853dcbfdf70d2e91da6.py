import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
        
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
        
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # 1. Strassen terms for j,k in {0,1}
    # P1 = A00(B00 + B11) -> C00 += 1, C11 += 1
    set_u(0, 0, 0, 1)
    set_v(0, 0, 0, 1)
    set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # P2 = (A00 + A10)B11 -> C00 -= 1, C10 += 1
    set_u(0, 0, 1, 1)
    set_u(1, 0, 1, 1)
    set_v(1, 1, 1, 1)
    set_w(0, 0, 1, -1)
    set_w(0, 1, 1, 1)

    # P3 = (A01 + A11)B00 -> C01 += 1, C11 -= 1
    set_u(0, 1, 2, 1)
    set_u(1, 1, 2, 1)
    set_v(0, 0, 2, 1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, -1)

    # P4 = A11(B00 - B11) -> C00 += 1, C10 += 1
    set_u(1, 1, 3, 1)
    set_v(0, 0, 3, 1)
    set_v(1, 1, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # P5 = (A00 + A01)B10 -> C01 += 1, C11 -= 1
    set_u(0, 0, 4, 1)
    set_u(0, 1, 4, 1)
    set_v(1, 0, 4, 1)
    set_w(1, 0, 4, 1)
    set_w(1, 1, 4, -1)

    # P6 = (A10 - A00)(B01 + B11) -> C00 += 1
    set_u(1, 0, 5, 1)
    set_u(0, 0, 5, -1)
    set_v(0, 1, 5, 1)
    set_v(1, 1, 5, 1)
    set_w(0, 0, 5, 1)

    # P7 = (A01 - A11)(B00 + B10) -> C11 += 1
    set_u(0, 1, 6, 1)
    set_u(1, 1, 6, -1)
    set_v(0, 0, 6, 1)
    set_v(1, 0, 6, 1)
    set_w(1, 1, 6, 1)

    # 2. Terms with j=2 (A_i2 B_2k)
    # i=0, k=0
    set_u(0, 2, 7, 1)
    set_v(2, 0, 7, 1)
    set_w(0, 0, 7, 1)

    # i=0, k=1
    set_u(0, 2, 8, 1)
    set_v(2, 1, 8, 1)
    set_w(1, 0, 8, 1)

    # i=0, k=2
    set_u(0, 2, 9, 1)
    set_v(2, 2, 9, 1)
    set_w(2, 0, 9, 1)

    # i=1, k=0
    set_u(1, 2, 10, 1)
    set_v(2, 0, 10, 1)
    set_w(0, 1, 10, 1)

    # i=1, k=1
    set_u(1, 2, 11, 1)
    set_v(2, 1, 11, 1)
    set_w(1, 1, 11, 1)

    # i=1, k=2
    set_u(1, 2, 12, 1)
    set_v(2, 2, 12, 1)
    set_w(2, 1, 12, 1)

    # 3. Terms with k=2, j in {0,1} (A_ij B_j2)
    # i=0, j=0
    set_u(0, 0, 13, 1)
    set_v(0, 2, 13, 1)
    set_w(2, 0, 13, 1)

    # i=0, j=1
    set_u(0, 1, 14, 1)
    set_v(1, 2, 14, 1)
    set_w(2, 0, 14, 1)

    # i=1, j=0
    set_u(1, 0, 15, 1)
    set_v(0, 2, 15, 1)
    set_w(2, 1, 15, 1)

    # i=1, j=1
    set_u(1, 1, 16, 1)
    set_v(1, 2, 16, 1)
    set_w(2, 1, 16, 1)

    return {"U": U, "V": V, "W": W}
