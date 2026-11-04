import time

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Strassen 2x2x2 coefficients
    S_U = [
        [1, 0, 0, 1],   # M1 = (A00+A11)(B00+B11)
        [0, 0, 1, 1],   # M2 = (A10+A11)B00
        [1, 0, 0, 0],   # M3 = A00(B10-B11)
        [0, 0, 0, 1],   # M4 = A11(B01-B10)
        [1, 1, 0, 0],   # M5 = (A00+A10)B11
        [-1, 0, 1, 0],  # M6 = (A11-A00)(B00+B10)
        [0, 1, 0, -1]   # M7 = (A10-A01)(B11+B01)
    ]
    
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks
    for block_idx in range(4):
        k_start = block_idx * 2
        
        for r in range(7):
            # U for A1: indices (i, a) where i in 0..1, a in 0..1
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] += val
            
            # V for B1 block: indices (a, k) where a in 0..1, k in k_start..k_start+1
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        v_idx = la * p + (k_start + lk)
                        V[v_idx][col] += val
            
            # W for C block: indices (k, i) where k in k_start..k_start+1, i in 0..1
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        w_idx = (k_start + lk) * n + li
                        W[w_idx][col] += val
            
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x8) - outer product
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2  # A[i, 2]
            v_idx = 2 * p + k  # B[2, k]
            w_idx = k * n + i  # C[k, i]
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
