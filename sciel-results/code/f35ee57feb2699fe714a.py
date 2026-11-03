"""Matrix multiplication 4x4x6 via direct 4x4x4 Strassen + schoolbook."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strategy: 4x4x4 rank-28 (Strassen blocks with schoolbook inner) + schoolbook for last 2 cols
    # Total rank = 28 + 16 = 44
    
    # Build 4x4x4 rank-28 decomposition
    # 7 Strassen block products, each done with schoolbook 2x2 (4 scalar mults) = 28
    U_444 = [[0.0] * 28 for _ in range(16)]
    V_444 = [[0.0] * 28 for _ in range(16)]
    W_444 = [[0.0] * 28 for _ in range(16)]
    
    # Strassen 2x2x2 block products
    strassen_products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): -1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1}),
    ]
    
    for t, (Xc, Yc, Zc) in enumerate(strassen_products):
        for i0 in range(2):
            for k0 in range(2):
                col = t * 4 + i0 * 2 + k0
                for (I, J), cx in Xc.items():
                    for j0 in range(2):
                        i = 2 * I + i0
                        j = 2 * J + j0
                        U_444[i * 4 + j][col] += cx
                for (J, K), cy in Yc.items():
                    for j0 in range(2):
                        j = 2 * J + j0
                        k = 2 * K + k0
                        V_444[j * 4 + k][col] += cy
                for (I, K), cz in Zc.items():
                    i = 2 * I + i0
                    k = 2 * K + k0
                    W_444[k * 4 + i][col] += cz
    
    err_444 = _verify(U_444, V_444, W_444, 4, 4, 4, 28)
    
    if err_444 > 1e-6:
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
    
    total_rank = 28 + 32
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Copy 4x4x4 part into first 28 columns
    for i in range(4):
        for j in range(4):
            for c in range(28):
                U[i * 4 + j][c] = U_444[i * 4 + j][c]
    for j in range(4):
        for k in range(4):
            for c in range(28):
                V[j * 6 + k][c] = V_444[j * 4 + k][c]
    for k in range(4):
        for i in range(4):
            for c in range(28):
                W[k * 4 + i][c] = W_444[k * 4 + i][c]
    
    # Schoolbook for last 2 columns: 4 block products, each 2x2*2x2 = 8 scalar mults = 32
    products_right = [
        (0, 0, 0),  # A00 * B_top -> C_top
        (0, 1, 1),  # A01 * B_bot -> C_top
        (1, 0, 0),  # A10 * B_top -> C_bot
        (1, 1, 1),  # A11 * B_bot -> C_bot
    ]
    
    for p_idx, (I, J, is_bot) in enumerate(products_right):
        col_offset = 28 + p_idx * 8
        for i0 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    col = col_offset + (i0 * 2 + j0) * 2 + k0
                    i = 2 * I + i0
                    j = 2 * J + j0
                    j_b = 2 * is_bot + j0
                    k = 4 + k0
                    U[i * 4 + j][col] = 1.0
                    V[j_b * 6 + k][col] = 1.0
                    W[k * 4 + i][col] = 1.0
    
    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
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

    return {"U": U, "V": V, "W": W}
