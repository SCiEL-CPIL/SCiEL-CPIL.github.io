import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS
    rank = 50

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # 1
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # 2
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # 3
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # 4
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # 5
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # 6
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        col += 1
        
        # 7
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        # 1
        U[a][col] += 1
        V[e][col] += 1
        W[c00][col] += 1
        col += 1
        
        # 2
        U[b][col] += 1
        V[g][col] += 1
        W[c00][col] += 1
        col += 1
        
        # 3
        U[c][col] += 1
        V[e][col] += 1
        W[c10][col] += 1
        col += 1
        
        # 4
        U[d][col] += 1
        V[g][col] += 1
        W[c10][col] += 1
        col += 1
        
        return True

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Left block of A (cols 0,1) with left block of B (rows 0,1)
    # Strassen for k in {0,1}, {2,3}, {4,5}
    for k0 in (0, 2, 4):
        if not add_strassen_222(0, 0, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Left block of A (cols 0,1) with rightmost col of B (row 6)
    if not add_schoolbook_221(0, 0, 6):
        pass
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Right block of A (cols 2,3) with left block of B (rows 2,3)
    # Strassen for k in {0,1}, {2,3}, {4,5}
    for k0 in (0, 2, 4):
        if not add_strassen_222(0, 2, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Right block of A (cols 2,3) with rightmost col of B (row 6)
    if not add_schoolbook_221(0, 2, 6):
        pass

    return {"U": U, "V": V, "W": W}
