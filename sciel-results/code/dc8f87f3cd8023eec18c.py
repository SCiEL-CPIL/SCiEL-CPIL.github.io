"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 3, 4

    # ---- Build a rank-7 Strassen decomposition of the 3x3x3 tensor ----
    # T3[i,j,k] = 1 if i==j==k. We need U3[i,c]*V3[j,c]*W3[k,c] summed over c == T3.
    U3 = np.zeros((n, 7), dtype=np.float64)
    V3 = np.zeros((m, 7), dtype=np.float64)
    W3 = np.zeros((n, 7), dtype=np.float64)
    # Standard Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7 ; C01 = M3+M5
    # C10 = M2+M4 ; C11 = M1-M2+M3+M6
    # A indices (row,col): 00=0, 01=1, 10=2, 11=3 ; B: 00=0, 01=1, 10=2, 11=3
    def setA(r, c, v): U3[r, c] = v
    def setB(r, c, v): V3[r, c] = v
    def setC(r, c, v): W3[r, c] = v
    # A factor entries per product
    setA(0, 0, 1); setA(3, 0, 1)                       # M1
    setA(2, 1, 1); setA(3, 1, 1)                       # M2
    setA(0, 2, 1)                                      # M3
    setA(3, 3, 1)                                      # M4
    setA(0, 4, 1); setA(1, 4, 1)                       # M5
    setA(2, 5, 1); setA(0, 5, -1)                      # M6
    setA(1, 6, 1); setA(3, 6, -1)                      # M7
    # B factor entries per product
    setB(0, 0, 1); setB(3, 0, 1)                       # M1
    setB(0, 1, 1)                                      # M2
    setB(2, 2, 1); setB(3, 2, -1)                      # M3
    setB(1, 3, 1); setB(2, 3, -1)                      # M4
    setB(3, 4, 1)                                      # M5
    setB(0, 5, 1); setB(1, 5, 1)                       # M6
    setB(2, 6, 1); setB(3, 6, 1)                       # M7
    # C (W) factor entries per product
    setC(0, 0, 1); setC(3, 0, 1)                       # M1
    setC(2, 1, -1); setC(3, 1, 1)                      # M2
    setC(0, 2, 1); setC(3, 2, 1)                       # M3
    setC(3, 3, 1); setC(0, 3, -1)                      # M4
    setC(0, 4, 1); setC(1, 4, 1)                       # M5
    setC(2, 5, 1); setC(3, 5, 1)                       # M6
    setC(0, 6, 1); setC(3, 6, -1)                      # M7

    # Verify the 3x3x3 tensor exactly (floating point, integer values).
    # T3[i,j,k] = sum_c U3[i,c]*V3[j,c]*W3[k,c] should equal 1 if i==j==k else 0.
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(n):
                s = 0.0
                for c in range(7):
                    s += U3[i, c] * V3[j, c] * W3[k, c]
                expected = 1.0 if (i == j == k) else 0.0
                err += abs(s - expected)
    if err > 1e-9:
        # Fallback to schoolbook if Strassen is somehow broken.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # ---- Tensor with I_3 to lift 3x3x3 -> 3x3x4 (rank 3*7 = 21) ----
    r3 = 7
    rank = 3 * r3
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    for c in range(r3):
        for t in range(3):  # t indexes the Kronecker (identity) dimension
            base = c * 3 + t
            for i in range(n):
                U[i * m + t, base] = U3[i, c]
            for j in range(m):
                V[j * p + t, base] = V3[j, c]
            for k in range(n):
                W[k * n + t, base] = W3[k, c]

    # Convert to lists of integer (or integer) entries.
    U_list = [[(int(round(x))) for x in row] for row in U]
    V_list = [[(int(round(x))) for x in row] for row in V]
    W_list = [[(int(round(x))) for x in row] for row in W]

    # Final exact verification against the 3x3x4 tensor.
    # T[i,j,k,l] = 1 if i==l else 0 (l in 0..3).
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(n):
                    s = 0
                    for c in range(rank):
                        s += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + l][c]
                    expected = 1 if (i == l) else 0
                    if s != expected:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback to schoolbook.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
