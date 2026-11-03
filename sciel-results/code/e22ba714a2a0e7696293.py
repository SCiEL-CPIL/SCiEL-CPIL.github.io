import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank: 1 (for j=0) + 5 * 7 (for j=1,2 using Strassen) = 36
    rank = 36
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # --- Part 1: j=0 ---
    # t = 0
    # U[i, 0] for i in 0..1
    # V[0, k] for k in 0..9
    # W[k, i] for k in 0..9, i in 0..1
    
    U[0 * m + 0, 0] = 1
    U[1 * m + 0, 0] = 1
    
    for k in range(p):
        V[0 * p + k, 0] = 1
        W[k * n + 0, 0] = 1
        W[k * n + 1, 0] = 1
        
    # --- Part 2: j in {1, 2} ---
    # 5 blocks of 2x2x2
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # A indices: (i, j_local) where j_local in {0,1} maps to j_global in {1,2}
    # B indices: (j_local, k_local) where k_local in {0,1} maps to k_global in {2s, 2s+1}
    # C indices: (i, k_local) maps to (i, k_global)
    
    # Coefficients for U (A), V (B), W (C)
    # u[i, j], v[j, k], w[i, k]
    
    strassen_u = [
        [[1, 0], [0, 1]], # M1: a+d
        [[0, 0], [1, 1]], # M2: c+d
        [[1, 0], [0, 0]], # M3: a
        [[0, 0], [0, 1]], # M4: d
        [[1, 1], [0, 0]], # M5: a+b
        [[0, 0], [1, -1]],# M6: c-a
        [[0, 1], [0, -1]] # M7: b-d
    ]
    
    strassen_v = [
        [[1, 1], [1, 1]], # M1: e+h
        [[1, 0], [0, 0]], # M2: e
        [[0, 1], [0, -1]],# M3: f-h
        [[0, -1], [1, 0]],# M4: g-e
        [[0, 0], [0, 1]], # M5: h
        [[1, 1], [0, 0]], # M6: e+f
        [[0, 0], [1, 1]]  # M7: g+h
    ]
    
    strassen_w = [
        [[1, 0], [0, 1]], # M1: C00 += 1, C11 += 1
        [[0, 0], [1, 0]], # M2: C10 += 1
        [[0, 1], [1, 0]], # M3: C01 += 1, C10 += 1
        [[1, 0], [0, 1]], # M4: C00 += 1, C11 += 1
        [[-1, 1], [0, 0]],# M5: C00 -= 1, C01 += 1
        [[0, 0], [0, 1]], # M6: C11 += 1
        [[1, 0], [0, 0]]  # M7: C00 += 1
    ]
    
    # Wait, let's re-verify Strassen W coefficients.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: C00=1, C01=0, C10=0, C11=1
    # M2: C00=0, C01=0, C10=1, C11=-1
    # M3: C00=0, C01=1, C10=1, C11=1
    # M4: C00=1, C01=0, C10=1, C11=1
    # M5: C00=-1, C01=1, C10=0, C11=0
    # M6: C00=0, C01=0, C10=0, C11=1
    # M7: C00=1, C01=0, C10=0, C11=0
    
    strassen_w = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, -1]],# M2
        [[0, 1], [1, 1]], # M3
        [[1, 0], [1, 1]], # M4
        [[-1, 1], [0, 0]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    t_start = 1
    for s in range(5):
        k_base = 2 * s
        for q in range(7):
            t = t_start + s * 7 + q
            
            # Fill U
            for i in range(2):
                for j_loc in range(2):
                    val = strassen_u[q][i][j_loc]
                    j_glob = 1 + j_loc
                    U[i * m + j_glob, t] = val
                    
            # Fill V
            for j_loc in range(2):
                for k_loc in range(2):
                    val = strassen_v[q][j_loc][k_loc]
                    j_glob = 1 + j_loc
                    k_glob = k_base + k_loc
                    V[j_glob * p + k_glob, t] = val
                    
            # Fill W
            for i in range(2):
                for k_loc in range(2):
                    val = strassen_w[q][i][k_loc]
                    k_glob = k_base + k_loc
                    W[k_glob * n + i, t] = val
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
