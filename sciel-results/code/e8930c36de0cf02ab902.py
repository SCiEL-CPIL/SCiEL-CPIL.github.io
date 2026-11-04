import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Rank 42 construction using Strassen for 2x2x2 blocks
    rank = 42
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)

    def add_strassen_block(i_start, j_start, k_start, offset):
        # Local indices
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)
        
        # Term 0 (M1): (A00+A11)(B00+B11)
        U[a00, offset] += 1
        U[a11, offset] += 1
        V[b00, offset] += 1
        V[b11, offset] += 1
        W[w00, offset] += 1
        W[w11, offset] += 1
        
        # Term 1 (M2): (A10+A11)B00
        U[a10, offset + 1] += 1
        U[a11, offset + 1] += 1
        V[b00, offset + 1] += 1
        W[w10, offset + 1] += 1
        W[w11, offset + 1] -= 1
        
        # Term 2 (M3): A00(B01-B11)
        U[a00, offset + 2] += 1
        V[b01, offset + 2] += 1
        V[b11, offset + 2] -= 1
        W[w01, offset + 2] += 1
        W[w11, offset + 2] += 1
        
        # Term 3 (M4): A11(B10-B00)
        U[a11, offset + 3] += 1
        V[b10, offset + 3] += 1
        V[b00, offset + 3] -= 1
        W[w00, offset + 3] += 1
        W[w10, offset + 3] += 1
        
        # Term 4 (M5): (A00+A01)B11
        U[a00, offset + 4] += 1
        U[a01, offset + 4] += 1
        V[b11, offset + 4] += 1
        W[w00, offset + 4] -= 1
        W[w01, offset + 4] += 1
        
        # Term 5 (M6): (A10-A00)(B00+B01)
        U[a10, offset + 5] += 1
        U[a00, offset + 5] -= 1
        V[b00, offset + 5] += 1
        V[b01, offset + 5] += 1
        W[w11, offset + 5] += 1
        
        # Term 6 (M7): (A01-A11)(B10+B11)
        U[a01, offset + 6] += 1
        U[a11, offset + 6] -= 1
        V[b10, offset + 6] += 1
        V[b11, offset + 6] += 1
        W[w00, offset + 6] += 1

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    # Verification
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True

    if not verify(U, V, W):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    
        return {"U": U, "V": V, "W": W}

    U_list = [[int(x) for x in row] for row in U.tolist()]
    V_list = [[int(x) for x in row] for row in V.tolist()]
    W_list = [[int(x) for x in row] for row in W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
