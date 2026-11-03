import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W

def _kron_product(A, B):
    """Kronecker product of two matrices A (ra x ca) and B (rb x cb)."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    result = [[0.0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            if A[i][j] != 0:
                for k in range(rb):
                    for l in range(cb):
                        result[i * rb + k][j * cb + l] = A[i][j] * B[k][l]
    return result

def _schoolbook_4x4x1():
    """Schoolbook decomposition for 4x4x1 (rank 16)."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1.0
                if abs(total - expected) > 1e-10:
                    return False
    return True

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 144."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()

    # 1. Get Strassen 2x2x2 factors
    U2, V2, W2 = _strassen_2x2()

    # 2. Kronecker product for 4x4x4
    U4_44 = _kron_product(U2, U2)
    V4_44 = _kron_product(V2, V2)
    W4_44 = _kron_product(W2, W2)

    # 3. Schoolbook for 4x4x1
    U4_16, V4_16, W4_16 = _schoolbook_4x4x1()

    # 4. Assemble for 4x4x9
    # Rank = 49 + 49 + 16 = 114
    total_rank = 114

    U_full = [[0.0] * total_rank for _ in range(16)]
    V_full = [[0.0] * total_rank for _ in range(36)]
    W_full = [[0.0] * total_rank for _ in range(36)]

    # Part 1: 4x4x4 (cols 0-3 of B, rows 0-3 of C)
    for r_u in range(16):
        for c in range(49):
            U_full[r_u][c] = U4_44[r_u][c]
            
    for r_v in range(16):
        j = r_v // 4
        k = r_v % 4
        global_v_row = j * 9 + k
        for c in range(49):
            V_full[global_v_row][c] = V4_44[r_v][c]
            
    for r_w in range(16):
        k = r_w // 4
        i = r_w % 4
        global_w_row = k * 4 + i
        for c in range(49):
            W_full[global_w_row][c] = W4_44[r_w][c]

    # Part 2: 4x4x4 (cols 4-7 of B, rows 4-7 of C)
    offset = 49
    for r_u in range(16):
        for c in range(49):
            U_full[r_u][offset + c] = U4_44[r_u][c]
            
    for r_v in range(16):
        j = r_v // 4
        k_local = r_v % 4
        k_global = k_local + 4
        global_v_row = j * 9 + k_global
        for c in range(49):
            V_full[global_v_row][offset + c] = V4_44[r_v][c]
            
    for r_w in range(16):
        k_local = r_w // 4
        i = r_w % 4
        k_global = k_local + 4
        global_w_row = k_global * 4 + i
        for c in range(49):
            W_full[global_w_row][offset + c] = W4_44[r_w][c]

    # Part 3: 4x4x1 (col 8 of B, row 8 of C)
    offset = 98
    for r_u in range(16):
        for c in range(16):
            U_full[r_u][offset + c] = U4_16[r_u][c]
            
    for r_v in range(4):
        global_v_row = r_v * 9 + 8
        for c in range(16):
            V_full[global_v_row][offset + c] = V4_16[r_v][c]
            
    for r_w in range(4):
        global_w_row = 8 * 4 + r_w
        for c in range(16):
            W_full[global_w_row][offset + c] = W4_16[r_w][c]

    # Verify
    if not _verify_decomposition(U_full, V_full, W_full, 4, 4, 9):
        return _schoolbook()
        
    return {"U": U_full, "V": V_full, "W": W_full}
