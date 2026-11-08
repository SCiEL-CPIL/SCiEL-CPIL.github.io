import time

import numpy as np

_DEADLINE = 120.0
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def construct():
    n, m, p = 2, 3, 9
    rank = 33

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    # Part 1: 4 blocks of 2x2x2 Strassen
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # Strassen rank 7
        # M1 = (A00 + A11)(B00 + B11)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        W[w_01][col] -= 1
        W[w_10][col] -= 1

        # M2 = (A10 + A11)B00
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        W[w_10][col+1] += 1
        W[w_11][col+1] += 1

        # M3 = A00(B10 - B11)
        U[u_a][col+2] += 1
        V[v_g][col+2] += 1
        V[v_h][col+2] -= 1
        W[w_00][col+2] += 1
        W[w_01][col+2] += 1

        # M4 = A11(B01 - B00)
        U[u_d][col+3] += 1
        V[v_f][col+3] += 1
        V[v_e][col+3] -= 1
        W[w_10][col+3] += 1
        W[w_11][col+3] += 1

        # M5 = (A00 + A01)B11
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        W[w_01][col+4] += 1
        W[w_11][col+4] += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        W[w_00][col+5] -= 1
        W[w_10][col+5] += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        W[w_01][col+6] += 1
        W[w_11][col+6] -= 1

        col += 7

    # Part 2: 1 block of 2x2x1 (Col 8 of B)
    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    # Rank 4 for 2x2x1
    # A00 B08
    U[u_a][col] += 1
    V[v_e8][col] += 1
    W[w_08][col] += 1

    # A01 B18
    U[u_b][col+1] += 1
    V[v_g8][col+1] += 1
    W[w_08][col+1] += 1

    # A10 B08
    U[u_c][col+2] += 1
    V[v_e8][col+2] += 1
    W[w_18][col+2] += 1

    # A11 B18
    U[u_d][col+3] += 1
    V[v_g8][col+3] += 1
    W[w_18][col+3] += 1

    col += 4

    # Part 3: A col 2 with B row 2
    # Rank 1
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    # U for A02, A12
    U[u_ar0][col] += 1
    U[u_ar1][col] += 1
    
    # V for B20..B28
    for k in range(p):
        v_bk = 2 * p + k
        V[v_bk][col] += 1
        
    # W for all C0j, C1j
    for k in range(p):
        for i in range(n):
            w_ki = k * n + i
            W[w_ki][col] += 1

    col += 1

    return {"U": U, "V": V, "W": W}
