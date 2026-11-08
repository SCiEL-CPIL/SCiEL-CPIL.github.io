import time
_DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s limit

def construct():
    n, m, p = _DIMENSIONS  # n=2, m=4, p=8
    # Deadline guard: if we've used too much time, return immediately
    if time.time() - _START_TIME > _TIMEOUT:
        # Fallback: return a valid but higher-rank decomposition
        rank = n * m * p  # 64
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m + j][r] = 1
                    V[j*p + k][r] = 1
                    W[k*n + i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen forms
    # X_forms[s][i*2+j'] for s=0..6, i=0,1, j'=0,1
    X_forms = [
        [1, 0, 0, 1],   # m1: (X00 + X11)
        [0, 0, 1, 1],   # m2: (X10 + X11)
        [1, 0, 0, 0],   # m3: X00
        [0, 0, 0, 1],   # m4: X11
        [1, 1, 0, 0],   # m5: (X00 + X01)
        [-1, 0, 1, 0],  # m6: (X10 - X00)
        [0, 1, 0, -1],  # m7: (X01 - X11)
    ]
    
    # Y_forms[s][j'*2+k'] for s=0..6, j'=0,1, k'=0,1
    Y_forms = [
        [1, 0, 0, 1],   # m1: (Y00 + Y11)
        [1, 0, 0, 0],   # m2: Y00
        [0, 1, 0, -1],  # m3: (Y01 - Y11)
        [-1, 0, 1, 0],  # m4: (Y10 - Y00)
        [0, 0, 0, 1],   # m5: Y11
        [1, 1, 0, 0],   # m6: (Y00 + Y01)
        [0, 0, 1, 1],   # m7: (Y10 + Y11)
    ]
    
    # Z_combos[z_idx][s] for z_idx=0..3 (z00,z01,z10,z11), s=0..6
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   # z00 = m1 + m4 - m5 + m7
        [0, 0, 1, 0, 1, 0, 0],    # z01 = m3 + m5
        [0, 1, 0, 1, 0, 0, 0],    # z10 = m2 + m4
        [1, -1, 1, 0, 0, 1, 0],   # z11 = m1 - m2 + m3 + m6
    ]
    
    num_groups = 4  # g = 0,1,2,3
    num_ablocks = 2  # b = 0,1 (A1, A2)
    num_strassen = 7  # s = 0..6
    rank = num_groups * num_ablocks * num_strassen  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                # U: for i in {0,1}, j' in {0,1}
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][r] = X_forms[s][i * 2 + jp]
                
                # V: for j' in {0,1}, k' in {0,1}
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r] = Y_forms[s][jp * 2 + kp]
                
                # W: for i in {0,1}, k' in {0,1}
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][r] = Z_combos[i * 2 + kp][s]
                
                r += 1
    
    return {"U": U, "V": V, "W": W}
