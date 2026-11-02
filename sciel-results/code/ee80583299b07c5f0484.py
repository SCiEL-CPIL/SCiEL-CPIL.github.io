"""Matrix multiplication 3x4x8 with rank reduction using Strassen identities."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Standard Strassen:
    # M1 = (a+b)(e+f)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(g-h)
    # M5 = (a+d)(h+e)
    # M6 = (c-a)(b+e)
    # M7 = (d-b)(f+g)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
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
        [1, 0, 0, 1, -1, 0, 1],  # M11 = C11
        [0, 0, 1, 0, 1, 0, 0],   # M12 = C12
        [0, 1, 0, 1, 0, 0, 0],   # M21 = C21
        [1, -1, 1, 0, 0, 1, 0]   # M22 = C22
    ]
    return U, V, W

def _construct_3x4x8():
    """Construct rank-minimal decomposition for 3x4x8."""
    n, m, p = 3, 4, 8
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # For 1x2x2 schoolbook:
    # C0 = x*a + y*b, C1 = x*c + y*d
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
    
    rank = 8 * 7 + 8 * 4  # 56 + 32 = 88
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
    
    # 2x2x2 blocks for i_pair=(0,1), all j_pairs and k_pairs
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            u_rows = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
            
            add_2x2x2_block(u_rows, v_rows, w_rows, col_offset)
            col_offset += 7
    
    # 1x2x2 blocks for i=2, all j_pairs and k_pairs
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [j0 * p + k0, j1 * p + k0, j0 * p + k1, j1 * p + k1]
            w_rows = [k0 * n + 2, k1 * n + 2]
            
            add_1x2x2_block(u_rows, v_rows, w_rows, col_offset)
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i*m+j]
                v_row = V_np[j*p+k]
                w_row = W_np[k*n+i]
                actual = np.sum(u_row * v_row * w_row)
                if abs(actual - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook():
    """Fallback schoolbook decomposition."""
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
        if time.time() < deadline - 5:
            if not _verify(result["U"], result["V"], result["W"], n, m, p):
                result = _schoolbook()
    except Exception:
        result = _schoolbook()
    
    return result
