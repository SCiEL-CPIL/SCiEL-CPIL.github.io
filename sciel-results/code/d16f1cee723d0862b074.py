import numpy as np

def _strassen_2x2_2cols(U, V, W, a_idx, b_idx, c_idx, d_idx, e_idx, f_idx, g_idx, h_idx, w00, w01, w10, w11, base_col):
    # M1 = (a+d)(e+h)
    U[a_idx][base_col] += 1
    U[d_idx][base_col] += 1
    V[e_idx][base_col] += 1
    V[h_idx][base_col] += 1
    # M2 = (c+d)(e)
    U[c_idx][base_col+1] += 1
    U[d_idx][base_col+1] += 1
    V[e_idx][base_col+1] += 1
    # M3 = (a)(f-h)
    U[a_idx][base_col+2] += 1
    V[f_idx][base_col+2] += 1
    V[h_idx][base_col+2] -= 1
    # M4 = (d)(g-e)
    U[d_idx][base_col+3] += 1
    V[g_idx][base_col+3] += 1
    V[e_idx][base_col+3] -= 1
    # M5 = (a+b)(h)
    U[a_idx][base_col+4] += 1
    U[b_idx][base_col+4] += 1
    V[h_idx][base_col+4] += 1
    # M6 = (c-a)(e+f)
    U[c_idx][base_col+5] += 1
    U[a_idx][base_col+5] -= 1
    V[e_idx][base_col+5] += 1
    V[f_idx][base_col+5] += 1
    # M7 = (b-d)(g+h)
    U[b_idx][base_col+6] += 1
    U[d_idx][base_col+6] -= 1
    V[g_idx][base_col+6] += 1
    V[h_idx][base_col+6] += 1
    # Reconstruct
    W[w00][base_col] += 1
    W[w00][base_col+3] += 1
    W[w00][base_col+4] -= 1
    W[w00][base_col+6] += 1
    W[w01][base_col+2] += 1
    W[w01][base_col+4] += 1
    W[w10][base_col+1] += 1
    W[w10][base_col+3] += 1
    W[w11][base_col] += 1
    W[w11][base_col+1] -= 1
    W[w11][base_col+2] += 1
    W[w11][base_col+5] += 1
    return base_col + 7

def _direct_2x2_1col(U, V, W, a_idx, b_idx, c_idx, d_idx, e_idx, g_idx, w0, w1, base_col):
    U[a_idx][base_col] += 1
    V[e_idx][base_col] += 1
    U[b_idx][base_col+1] += 1
    V[g_idx][base_col+1] += 1
    U[c_idx][base_col+2] += 1
    V[e_idx][base_col+2] += 1
    U[d_idx][base_col+3] += 1
    V[g_idx][base_col+3] += 1
    W[w0][base_col] += 1
    W[w0][base_col+1] += 1
    W[w1][base_col+2] += 1
    W[w1][base_col+3] += 1
    return base_col + 4

def _direct_1x1_1col(U, V, W, a_idx, b_idx, w_idx, base_col):
    U[a_idx][base_col] += 1
    V[b_idx][base_col] += 1
    W[w_idx][base_col] += 1
    return base_col + 1

def construct():
    n, m, p = 2, 3, 9
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # A indices
    ai00 = 0; ai01 = 1; ai10 = 3; ai11 = 4
    ai02 = 2; ai12 = 5
    # B indices
    b0k = lambda k: k
    b1k = lambda k: 9 + k
    b2k = lambda k: 18 + k
    # C indices
    ck = lambda k: k * n
    
    for grp in range(4):
        k0 = 2 * grp
        k1 = 2 * grp + 1
        col = _strassen_2x2_2cols(
            U, V, W,
            ai00, ai01, ai10, ai11,
            b0k(k0), b0k(k1), b1k(k0), b1k(k1),
            ck(k0)+0, ck(k1)+0, ck(k0)+1, ck(k1)+1,
            col
        )
    
    col = _direct_2x2_1col(
        U, V, W,
        ai00, ai01, ai10, ai11,
        b0k(8), b1k(8),
        ck(8)+0, ck(8)+1,
        col
    )
    
    for kk in range(9):
        col = _direct_1x1_1col(U, V, W, ai02, b2k(kk), ck(kk)+0, col)
        col = _direct_1x1_1col(U, V, W, ai12, b2k(kk), ck(kk)+1, col)
    
    return {"U": U, "V": V, "W": W}
