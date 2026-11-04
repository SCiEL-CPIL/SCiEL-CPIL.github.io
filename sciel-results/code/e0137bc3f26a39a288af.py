import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col):
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1

    # M1
    U[i0*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M2
    U[i1*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    W[k0*n+i1][col] = 1
    W[k1*n+i1][col] = -1
    col += 1

    # M3
    U[i0*m+j0][col] = 1
    V[j0*p+k1][col] = 1
    V[j1*p+k1][col] = -1
    W[k1*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M4
    U[i1*m+j1][col] = 1
    V[j1*p+k0][col] = 1
    V[j0*p+k0][col] = -1
    W[k0*n+i0][col] = 1
    W[k0*n+i1][col] = 1
    col += 1

    # M5
    U[i0*m+j0][col] = 1
    U[i0*m+j1][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = -1
    W[k1*n+i0][col] = 1
    col += 1

    # M6
    U[i1*m+j0][col] = 1
    U[i0*m+j0][col] = -1
    V[j0*p+k0][col] = 1
    V[j0*p+k1][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M7
    U[i0*m+j1][col] = 1
    U[i1*m+j1][col] = -1
    V[j1*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    col += 1

    return col

def _laderman_3x3(i_start, j_start, k_start, n, m, p, U, V, W, col):
    i0, i1, i2 = i_start, i_start+1, i_start+2
    j0, j1, j2 = j_start, j_start+1, j_start+2
    k0, k1, k2 = k_start, k_start+1, k_start+2

    # M1
    U[i0*m+j0][col] = 1
    U[i2*m+j2][col] = 1
    V[j0*p+k0][col] = 1
    V[j2*p+k2][col] = 1
    W[k0*n+i0][col] = 1
    W[k2*n+i2][col] = 1
    col += 1

    # M2
    U[i1*m+j1][col] = 1
    U[i2*m+j2][col] = 1
    V[j1*p+k1][col] = 1
    V[j2*p+k2][col] = 1
    W[k1*n+i1][col] = 1
    W[k2*n+i2][col] = 1
    col += 1

    # M3
    U[i0*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M4
    U[i2*m+j2][col] = 1
    U[i0*m+j0][col] = 1
    V[j2*p+k2][col] = 1
    V[j0*p+k0][col] = 1
    W[k2*n+i2][col] = 1
    W[k0*n+i0][col] = 1
    col += 1

    # M5
    U[i2*m+j2][col] = 1
    U[i1*m+j1][col] = 1
    V[j2*p+k2][col] = 1
    V[j1*p+k1][col] = 1
    W[k2*n+i2][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M6
    U[i0*m+j1][col] = 1
    U[i2*m+j1][col] = 1
    V[j1*p+k0][col] = 1
    V[j2*p+k0][col] = 1
    W[k0*n+i0][col] = 1
    W[k0*n+i2][col] = 1
    col += 1

    # M7
    U[i1*m+j0][col] = 1
    U[i1*m+j2][col] = 1
    V[j0*p+k1][col] = 1
    V[j2*p+k1][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    return col

def _verify_decomposition(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Partition 2+3
    n_parts = [2, 3]
    m_parts = [2, 3]
    p_parts = [2, 3]
    
    n_starts = [0, 2]
    m_starts = [0, 2]
    p_starts = [0, 2]
    
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni, mj, pk = n_parts[i], m_parts[j], p_parts[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                elif ni == 3 and mj == 3 and pk == 3:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    col = _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col)
                elif ni == 3 and mj == 3 and pk == 3:
                    col = _laderman_3x3(i_start, j_start, k_start, n, m, p, U, V, W, col)
                else:
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    # Fallback: Direct
    total_rank_direct = n * m * p
    U2 = [[0] * total_rank_direct for _ in range(n * m)]
    V2 = [[0] * total_rank_direct for _ in range(m * p)]
    W2 = [[0] * total_rank_direct for _ in range(p * n)]
    col2 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j][col2] = 1
                V2[j*p+k][col2] = 1
                W2[k*n+i][col2] = 1
                col2 += 1
    return {"U": U2, "V": V2, "W": W2}
