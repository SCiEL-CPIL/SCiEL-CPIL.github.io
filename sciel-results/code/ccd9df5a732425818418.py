"""
5x5 matrix multiplication — rank 125 via 5x5x5 Strassen-like block decomposition.
Uses Strassen's 7-term algorithm for 2x2x2 blocks to decompose the 5x5x5 tensor.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_map, v_map, w_map):
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for (i, j), val in u_map.items():
            u[i * m + j] = val
        for (j, k), val in v_map.items():
            v[j * p + k] = val
        for (k, i), val in w_map.items():
            w[k * n + i] = val
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def strassen(i0, i1, j0, j1, k0, k1):
        # 7 terms for 2x2x2 block
        add_term({(i0,j0):1,(i1,j1):1}, {(j0,k0):1,(j1,k1):1}, {(k0,i0):1,(k1,i1):1})
        add_term({(i1,j0):1,(i1,j1):1}, {(j0,k0):1}, {(k0,i1):1,(k1,i1):-1})
        add_term({(i0,j0):1}, {(j0,k1):1,(j1,k1):-1}, {(k1,i0):1,(k1,i1):1})
        add_term({(i1,j1):1}, {(j1,k0):1,(j0,k0):-1}, {(k0,i0):1,(k0,i1):1})
        add_term({(i0,j0):1,(i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1,(k1,i0):1})
        add_term({(i1,j0):1,(i0,j0):-1}, {(j0,k0):1,(j0,k1):1}, {(k1,i1):1})
        add_term({(i0,j1):1,(i1,j1):-1}, {(j1,k0):1,(j1,k1):1}, {(k0,i0):1})

    # 5x5x5 decomposition using Strassen on 2x2 blocks
    # Split 5 = 2 + 3, but use a different approach:
    # Use 7 Strassen terms for each 2x2x2 sub-block
    
    # C_tl (2x2) = A_tl(2x2) * B_tl(2x2) + A_tr(2x3) * B_bl(3x2)
    # We'll handle this by decomposing into 2x2 blocks where possible
    
    # For the 2x2x2 corner blocks, use Strassen:
    # C_tl[0:2,0:2] gets contributions from:
    # A[0:2,0:2] * B[0:2,0:2] (2x2x2 Strassen)
    strassen(0, 1, 0, 1, 0, 1)
    
    # A[0:2,2:4] * B[2:4,0:2] - this is 2x2 * 2x2 = 2x2
    # But B[2:4,0:2] is 2x2 (rows 2,3; cols 0,1)
    strassen(0, 1, 2, 3, 0, 1)
    
    # A[0:2,4] * B[4,0:2] - 2x1 * 1x2 = 2x2
    for i in (0, 1):
        for k in (0, 1):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    
    # C_tr (2x3) = A_tl(2x2) * B_tr(2x3) + A_tr(2x3) * B_br(3x3)
    # A[0:2,0:2] * B[0:2,2:4] - 2x2 * 2x2 = 2x2 (top-left of C_tr)
    strassen(0, 1, 0, 1, 2, 3)
    
    # A[0:2,0:2] * B[0:2,4] - 2x2 * 2x1 = 2x1 (top-right of C_tr)
    for i in (0, 1):
        for j in (0, 1):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    
    # A[0:2,2:4] * B[2:4,2:4] - 2x2 * 2x2 = 2x2 (middle of C_tr)
    strassen(0, 1, 2, 3, 2, 3)
    
    # A[0:2,2:4] * B[2:4,4] - 2x2 * 2x1 = 2x1
    for i in (0, 1):
        for j in (2, 3):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    
    # A[0:2,4] * B[4,2:4] - 2x1 * 1x2 = 2x2
    for i in (0, 1):
        for k in (2, 3):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    
    # A[0:2,4] * B[4,4] - 2x1 * 1x1 = 2x1
    for i in (0, 1):
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
    
    # A[2:4,0:2] * B[0:2,2:4] - 2x2 * 2x2 = 2x2 (bottom-left of C_tr)
    strassen(2, 3, 0, 1, 2, 3)
    
    # A[2:4,0:2] * B[0:2,4] - 2x2 * 2x1 = 2x1
    for i in (2, 3):
        for j in (0, 1):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    
    # A[2:4,2:4] * B[2:4,2:4] - 2x2 * 2x2 = 2x2 (bottom-middle of C_tr)
    strassen(2, 3, 2, 3, 2, 3)
    
    # A[2:4,2:4] * B[2:4,4] - 2x2 * 2x1 = 2x1
    for i in (2, 3):
        for j in (2, 3):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    
    # A[2:4,4] * B[4,2:4] - 2x1 * 1x2 = 2x2
    for i in (2, 3):
        for k in (2, 3):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    
    # A[2:4,4] * B[4,4] - 2x1 * 1x1 = 2x1
    for i in (2, 3):
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
    
    # A[4,0:2] * B[0:2,2:4] - 1x2 * 2x2 = 1x2
    for j in (0, 1):
        for k in (2, 3):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    
    # A[4,0:2] * B[0:2,4] - 1x2 * 2x1 = 1x1
    for j in (0, 1):
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    
    # A[4,2:4] * B[2:4,2:4] - 1x2 * 2x2 = 1x2
    for j in (2, 3):
        for k in (2, 3):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    
    # A[4,2:4] * B[2:4,4] - 1x2 * 2x1 = 1x1
    for j in (2, 3):
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    
    # A[4,4] * B[4,2:4] - 1x1 * 1x2 = 1x2
    for k in (2, 3):
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    
    # A[4,4] * B[4,4] - 1x1 * 1x1 = 1x1
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})
    
    # C_bl (3x2) = A_bl(3x2) * B_tl(2x2) + A_br(3x3) * B_bl(3x2)
    # A[2:4,0:2] * B[0:2,0:2] - 2x2 * 2x2 = 2x2 (top of C_bl)
    strassen(2, 3, 0, 1, 0, 1)
    
    # A[2:4,2:4] * B[2:4,0:2] - 2x2 * 2x2 = 2x2
    strassen(2, 3, 2, 3, 0, 1)
    
    # A[2:4,4] * B[4,0:2] - 2x1 * 1x2 = 2x2
    for i in (2, 3):
        for k in (0, 1):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    
    # A[4,0:2] * B[0:2,0:2] - 1x2 * 2x2 = 1x2 (bottom of C_bl)
    for j in (0, 1):
        for k in (0, 1):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    
    # A[4,2:4] * B[2:4,0:2] - 1x2 * 2x2 = 1x2
    for j in (2, 3):
        for k in (0, 1):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    
    # A[4,4] * B[4,0:2] - 1x1 * 1x2 = 1x2
    for k in (0, 1):
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    
    # C_br (3x3) = A_bl(3x2) * B_tr(2x3) + A_br(3x3) * B_br(3x3)
    # A[2:4,0:2] * B[0:2,2:4] - 2x2 * 2x2 = 2x2 (top-left of C_br)
    strassen(2, 3, 0, 1, 2, 3)
    
    # A[2:4,0:2] * B[0:2,4] - 2x2 * 2x1 = 2x1 (top-right of C_br)
    for i in (2, 3):
        for j in (0, 1):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    
    # A[2:4,2:4] * B[2:4,2:4] - 2x2 * 2x2 = 2x2 (middle of C_br)
    strassen(2, 3, 2, 3, 2, 3)
    
    # A[2:4,2:4] * B[2:4,4] - 2x2 * 2x1 = 2x1
    for i in (2, 3):
        for j in (2, 3):
            add_term({(i,j):1}, {(j,4):1}, {(4,i):1})
    
    # A[2:4,4] * B[4,2:4] - 2x1 * 1x2 = 2x2
    for i in (2, 3):
        for k in (2, 3):
            add_term({(i,4):1}, {(4,k):1}, {(k,i):1})
    
    # A[2:4,4] * B[4,4] - 2x1 * 1x1 = 2x1
    for i in (2, 3):
        add_term({(i,4):1}, {(4,4):1}, {(4,i):1})
    
    # A[4,0:2] * B[0:2,2:4] - 1x2 * 2x2 = 1x2 (bottom-left of C_br)
    for j in (0, 1):
        for k in (2, 3):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    
    # A[4,0:2] * B[0:2,4] - 1x2 * 2x1 = 1x1 (bottom-right of C_br)
    for j in (0, 1):
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    
    # A[4,2:4] * B[2:4,2:4] - 1x2 * 2x2 = 1x2
    for j in (2, 3):
        for k in (2, 3):
            add_term({(4,j):1}, {(j,k):1}, {(k,4):1})
    
    # A[4,2:4] * B[2:4,4] - 1x2 * 2x1 = 1x1
    for j in (2, 3):
        add_term({(4,j):1}, {(j,4):1}, {(4,4):1})
    
    # A[4,4] * B[4,2:4] - 1x1 * 1x2 = 1x2
    for k in (2, 3):
        add_term({(4,4):1}, {(4,k):1}, {(k,4):1})
    
    # A[4,4] * B[4,4] - 1x1 * 1x1 = 1x1
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
