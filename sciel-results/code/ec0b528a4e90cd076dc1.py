import time

DIMENSIONS = [2, 2, 3]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    The minimal rank for this tensor is 11.
    """
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # 1. (A00+A11)(B00+B11) -> M1
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    # W contributions for M1: C00+=1, C11+=1
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # 2. (A10+A11)B00 -> M2
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    # W: C10+=1, C11+=-1
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # 3. A00(B01-B11) -> M3
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    # W: C01+=1, C11+=1
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # 4. A11(B10-B00) -> M4
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    # W: C00+=1, C10+=1
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # 5. (A00+A01)B11 -> M5
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    # W: C00+=-1, C01+=1
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # 6. (A10-A00)(B00+B01) -> M6
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    # W: C11+=1
    set_w(1, 1, 5, 1)

    # 7. (A01-A11)(B10+B11) -> M7
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    # W: C00+=1
    set_w(0, 0, 6, 1)

    # 8. A00 B02 -> M8
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    # W: C02+=1
    set_w(2, 0, 7, 1)

    # 9. A01 B12 -> M9
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    # W: C02+=1
    set_w(2, 0, 8, 1)

    # 10. A10 B02 -> M10
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    # W: C12+=1
    set_w(2, 1, 9, 1)

    # 11. A11 B12 -> M11
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    # W: C12+=1
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
