import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = 2, 4, 8
    rank = 56

    U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 16 rows

    col = 0

    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            if _time_left() < 2.0:
                break

            j1 = j0 + 1
            k1 = k0 + 1

            # M1 = (A_{0,j0} + A_{1,j1}) * (B_{j0,k0} + B_{j1,k1})
            U[0*m + j0][col] = 1.0
            U[1*m + j1][col] = 1.0
            V[j0*p + k0][col] = 1.0
            V[j1*p + k1][col] = 1.0
            W[k0*n + 0][col] = 1.0
            W[k1*n + 1][col] = 1.0
            col += 1

            # M2 = (A_{1,j0} + A_{1,j1}) * B_{j0,k0}
            U[1*m + j0][col] = 1.0
            U[1*m + j1][col] = 1.0
            V[j0*p + k0][col] = 1.0
            W[k0*n + 1][col] = 1.0
            W[k1*n + 1][col] = -1.0
            col += 1

            # M3 = A_{0,j0} * (B_{j0,k1} - B_{j1,k1})
            U[0*m + j0][col] = 1.0
            V[j0*p + k1][col] = 1.0
            V[j1*p + k1][col] = -1.0
            W[k1*n + 0][col] = 1.0
            W[k1*n + 1][col] = 1.0
            col += 1

            # M4 = A_{1,j1} * (B_{j1,k0} - B_{j0,k0})
            U[1*m + j1][col] = 1.0
            V[j1*p + k0][col] = 1.0
            V[j0*p + k0][col] = -1.0
            W[k0*n + 0][col] = 1.0
            W[k0*n + 1][col] = 1.0
            col += 1

            # M5 = (A_{0,j0} + A_{0,j1}) * B_{j1,k1}
            U[0*m + j0][col] = 1.0
            U[0*m + j1][col] = 1.0
            V[j1*p + k1][col] = 1.0
            W[k0*n + 0][col] = -1.0
            W[k1*n + 0][col] = 1.0
            col += 1

            # M6 = (A_{1,j0} - A_{0,j0}) * (B_{j0,k0} + B_{j0,k1})
            U[1*m + j0][col] = 1.0
            U[0*m + j0][col] = -1.0
            V[j0*p + k0][col] = 1.0
            V[j0*p + k1][col] = 1.0
            W[k1*n + 1][col] = 1.0
            col += 1

            # M7 = (A_{0,j1} - A_{1,j1}) * (B_{j1,k0} + B_{j1,k1})
            U[0*m + j1][col] = 1.0
            U[1*m + j1][col] = -1.0
            V[j1*p + k0][col] = 1.0
            V[j1*p + k1][col] = 1.0
            W[k0*n + 0][col] = 1.0
            col += 1

    return {"U": U, "V": V, "W": W}
