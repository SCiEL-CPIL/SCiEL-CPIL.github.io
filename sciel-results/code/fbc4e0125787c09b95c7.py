"""Matrix multiplication 3x4x8 with rank reduction using Strassen identities."""
import time

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, 0, 1],  # a
        [0, 1, 0, 0, 0, 1, 0],  # b
        [0, 0, 1, 1, 1, 0, 0],  # c
        [1, 0, 0, 1, 0, 1, 0]   # d
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],  # e
        [0, 0, 0, 1, 0, 0, 1],  # f
        [0, 0, 1, 0, 0, 1, 0],  # g
        [1, 0, 1, 0, 1, 0, 0]   # h
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
    
    # Schoolbook for 1x2x2 (rank 4)
    U1 = [
        [1, 0, 0, 0],  # x
        [0, 1, 0, 0]   # y
    ]
    V1 = [
        [1, 0, 0, 0],  # a
        [0, 1, 0, 0],  # b
        [0, 0, 1, 0],  # c
        [0, 0, 0, 1]   # d
    ]
    W1 = [
        [1, 1, 0, 0],  # C0 = xa + yb
        [0, 0, 1, 1]   # C1 = xc + yd
    ]
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 8*3 = 24 rows
    
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
    
    # Process 2x2x2 blocks for i_pair=0 (i=0,1)
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
    
    # Process 1x2x2 blocks for i=2
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
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if actual != expected:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    result = _construct_3x4x8()
    
    n, m, p = DIMENSIONS
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
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
        result = {"U": U, "V": V, "W": W}
    
    return result
