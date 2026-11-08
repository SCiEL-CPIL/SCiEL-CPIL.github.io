import numpy as np
import time

_T0 = time.time()
_TL = 110.0

def construct():
    n, m, p = 2, 3, 9
    R = 50
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    col = 0
    for blk in range(4):
        c0 = 2 * blk
        c1 = c0 + 1
        v00 = c0
        v01 = c1
        v10 = p + c0
        v11 = p + c1
        a00, a01, a10, a11 = 0, 1, 3, 4

        U[a00][col] += 1; U[a11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        U[a10][col+1] += 1; U[a11][col+1] += 1
        V[v00][col+1] += 1
        U[a00][col+2] += 1
        V[v01][col+2] += 1; V[v11][col+2] -= 1
        U[a11][col+3] += 1
        V[v10][col+3] += 1; V[v00][col+3] -= 1
        U[a00][col+4] += 1; U[a01][col+4] += 1
        V[v11][col+4] += 1
        U[a10][col+5] += 1; U[a00][col+5] -= 1
        V[v00][col+5] += 1; V[v01][col+5] += 1
        U[a01][col+6] += 1; U[a11][col+6] -= 1
        V[v10][col+6] += 1; V[v11][col+6] += 1

        w00 = c0 * n
        w01 = c1 * n
        w10 = c0 * n + 1
        w11 = c1 * n + 1
        W[w00][col] += 1; W[w00][col+3] += 1; W[w00][col+4] -= 1; W[w00][col+6] += 1
        W[w01][col+2] += 1; W[w01][col+4] += 1
        W[w10][col+1] += 1; W[w10][col+3] += 1
        W[w11][col] += 1; W[w11][col+1] -= 1; W[w11][col+2] += 1; W[w11][col+5] += 1
        col += 7

    k8 = 8
    U[0][col] += 1; V[k8][col] += 1; W[k8 * n][col] += 1
    U[1][col+1] += 1; V[p + k8][col+1] += 1; W[k8 * n][col+1] += 1
    U[3][col+2] += 1; V[k8][col+2] += 1; W[k8 * n + 1][col+2] += 1
    U[4][col+3] += 1; V[p + k8][col+3] += 1; W[k8 * n + 1][col+3] += 1
    col += 4

    for k in range(9):
        U[2][col] += 1; V[2 * p + k][col] += 1; W[k * n][col] += 1
        U[5][col+1] += 1; V[2 * p + k][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2

    return {"U": U, "V": V, "W": W}
