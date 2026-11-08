import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check non-zero entries (i,j,k) -> must sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    # Check zero entries
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j2 = b // p
            k = b % p
            if j != j2:
                continue
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k != k2 or i != i2:
                    s = 0.0
                    for t in range(rank):
                        s += U[a][t] * V[b][t] * W[c][t]
                    if abs(s) > tol:
                        return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110

    # Strassen's 7-term decomposition for 2x2 * 2x2
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # M1: (A00+A11)(B00+B11)
    U1 = {(0,0): 1, (1,1): 1}
    V1 = {(0,0): 1, (1,1): 1}
    W1 = {(0,0): 1, (1,1): 1}
    # M2: (A10+A11)B00
    U2 = {(1,0): 1, (1,1): 1}
    V2 = {(0,0): 1}
    W2 = {(0,1): 1, (1,1): -1}
    # M3: A00(B10-B00)
    U3 = {(0,0): 1}
    V3 = {(1,0): 1, (0,0): -1}
    W3 = {(0,0): 1, (1,0): 1}
    # M4: A11(B01-B11)
    U4 = {(1,1): 1}
    V4 = {(0,1): 1, (1,1): -1}
    W4 = {(0,0): 1, (0,1): 1}
    # M5: (A00+A01)B11
    U5 = {(0,0): 1, (0,1): 1}
    V5 = {(1,1): 1}
    W5 = {(0,0): -1, (1,0): 1}
    # M6: (A10-A00)(B00+B01)
    U6 = {(1,0): 1, (0,0): -1}
    V6 = {(0,0): 1, (0,1): 1}
    W6 = {(1,1): 1}
    # M7: (A01-A11)(B10+B11)
    U7 = {(0,1): 1, (1,1): -1}
    V7 = {(1,0): 1, (1,1): 1}
    W7 = {(0,1): 1}

    strassen_terms = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]

    # Build the rank-28 block Strassen decomposition
    # Split the middle dimension (size 4) into two halves: j in {0,1} and j in {2,3}
    # This gives 4 independent 2x2 * 2x2 multiplications:
    # Block 0: j in {0,1}, k in {0,1}
    # Block 1: j in {0,1}, k in {2,3}
    # Block 2: j in {2,3}, k in {0,1}
    # Block 3: j in {2,3}, k in {2,3}

    rank_strassen = 28
    U = [[0] * rank_strassen for _ in range(n * m)]
    V = [[0] * rank_strassen for _ in range(m * p)]
    W = [[0] * rank_strassen for _ in range(p * n)]

    col = 0
    for block in range(4):
        j_offset = (block // 2) * 2
        k_offset = (block % 2) * 2

        for U_loc, V_loc, W_loc in strassen_terms:
            for (i_loc, j_loc), val in U_loc.items():
                i = i_loc
                j = j_offset + j_loc
                global_u_idx = i * m + j
                U[global_u_idx][col] += val

            for (j_loc, k_loc), val in V_loc.items():
                j = j_offset + j_loc
                k = k_offset + k_loc
                global_v_idx = j * p + k
                V[global_v_idx][col] += val

            for (k_loc, i_loc), val in W_loc.items():
                k = k_offset + k_loc
                i = i_loc
                global_w_idx = k * n + i
                W[global_w_idx][col] += val

            col += 1

    # Verify the Strassen decomposition
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 32
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j][col] = 1
                V_school[j*p + k][col] = 1
                W_school[k*n + i][col] = 1
                col += 1

    return {"U": U_school, "V": V_school, "W": W_school}
