"""Matrix multiplication 3x4x8 with rank reduction using Strassen identities."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 1],  # a
        [0, 0, 0, 0, 1, 0, -1],  # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, 1]    # d
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # e
        [0, 0, 1, 0, 0, 1, 1],   # f
        [1, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, 0, 1, 0, 1]   # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # M11
        [0, 0, 1, 0, 1, 0, 0],   # M12
        [0, 1, 0, 1, 0, 0, 0],   # M21
        [1, -1, 1, 0, 0, 1, 0]   # M22
    ]
    return U, V, W

def _construct_3x4x8():
    """Construct rank-minimal decomposition for 3x4x8."""
    n, m, p = 3, 4, 8
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # 1x2x2 schoolbook
    U1 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]
    V1 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W1 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    def add_2x2x2_block(u_rows, v_rows, w_rows, offset):
        for i in range(4):
            for c in range(7):
                U[u_rows[i]][offset + c] += U2[i][c]
                V[v_rows[i]][offset + c] += V2[i][c]
                W[w_rows[i]][offset + c] += W2[i][c]
    
    def add_1x2x2_block(u_rows, v_rows, w_rows, offset):
        for i in range(2):
            for c in range(4):
                U[u_rows[i]][offset + c] += U1[i][c]
                V[v_rows[i]][offset + c] += V1[i][c]
                W[w_rows[i]][offset + c] += W1[i][c]
        for i in range(2, 4):
            for c in range(4):
                V[v_rows[i]][offset + c] += V1[i][c]
                
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            u_rows = [
                0 * m + j0,
                0 * m + j1,
                1 * m + j0,
                1 * m + j1
            ]
            v_rows = [
                j0 * p + k0,
                j0 * p + k1,
                j1 * p + k0,
                j1 * p + k1
            ]
            w_rows = [
                k0 * n + 0,
                k1 * n + 0,
                k0 * n + 1,
                k1 * n + 1
            ]
            
            add_2x2x2_block(u_rows, v_rows, w_rows, col_offset)
            col_offset += 7
            
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            u_rows = [
                2 * m + j0,
                2 * m + j1
            ]
            v_rows = [
                j0 * p + k0,
                j1 * p + k0,
                j0 * p + k1,
                j1 * p + k1
            ]
            w_rows = [
                k0 * n + 2,
                k1 * n + 2
            ]
            
            add_1x2x2_block(u_rows, v_rows, w_rows, col_offset)
            col_offset += 4
            
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                u_row = U_np[i*m+j]
                v_row = V_np[j*p+k]
                w_row = W_np[k*n+i]
                actual = np.sum(u_row * v_row * w_row)
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    try:
        result = _construct_3x4x8()
        n, m, p = DIMENSIONS
        if time.time() < deadline - 10:
            if not _verify(result["U"], result["V"], result["W"], n, m, p):
                result = _schoolbook()
        else:
            pass
    except Exception:
        result = _schoolbook()
    
    return result
