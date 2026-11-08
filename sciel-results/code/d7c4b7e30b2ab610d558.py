import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen 2x2x2 multiplications
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Define the 4 sub-multiplications
    # m_idx: 0->P11, 1->P12, 2->P21, 3->P22
    # A_indices: indices in U (i*m + j)
    # B_indices: indices in V (j*p + k)
    # C_indices: indices in W (k*n + i) for P_00, P_01, P_10, P_11
    
    # P11: A_L (cols 0,1) * B_11 (rows 0,1) -> C_L (cols 0,1)
    # A_L: i=0,1; j=0,1. U idx: 0,1,4,5.
    # B_11: j=0,1; k=0,1. V idx: 0,1,4,5.
    # C_L: k=0,1; i=0,1. W idx: 0,2,1,3. (k*n+i)
    # P_00 -> C_0,0 (w=0), P_01 -> C_0,1 (w=2), P_10 -> C_1,0 (w=1), P_11 -> C_1,1 (w=3)
    
    configs = [
        {
            'A': [0, 1, 4, 5], # a, b, c, d
            'B': [0, 1, 4, 5], # x, y, z, w
            'C': [0, 2, 1, 3]  # P00, P01, P10, P11
        },
        {
            'A': [0, 1, 4, 5], # a, b, c, d
            'B': [2, 3, 6, 7], # x, y, z, w
            'C': [4, 6, 5, 7]  # P00, P01, P10, P11
        },
        {
            'A': [2, 3, 6, 7], # a, b, c, d
            'B': [8, 9, 12, 13], # x, y, z, w
            'C': [0, 2, 1, 3]  # P00, P01, P10, P11
        },
        {
            'A': [2, 3, 6, 7], # a, b, c, d
            'B': [10, 11, 14, 15], # x, y, z, w
            'C': [4, 6, 5, 7]  # P00, P01, P10, P11
        }
    ]
    
    # Strassen steps:
    # 0: M1 = (a+d)(x+w) -> P00:1, P11:1
    # 1: M2 = (c+d)x     -> P10:1
    # 2: M3 = a(y-w)     -> P01:1, P11:1
    # 3: M4 = d(z-x)     -> P00:1, P10:1
    # 4: M5 = (a+b)w     -> P00:-1, P01:1
    # 5: M6 = (c-a)(x+y) -> P11:1
    # 6: M7 = (b-d)(z+w) -> P00:1
    
    for m_idx, cfg in enumerate(configs):
        A_idx = cfg['A']
        B_idx = cfg['B']
        C_idx = cfg['C']
        
        # a, b, c, d indices in A_idx
        a, b, c, d = A_idx[0], A_idx[1], A_idx[2], A_idx[3]
        # x, y, z, w indices in B_idx
        x, y, z, w = B_idx[0], B_idx[1], B_idx[2], B_idx[3]
        # P00, P01, P10, P11 indices in C_idx
        p00, p01, p10, p11 = C_idx[0], C_idx[1], C_idx[2], C_idx[3]
        
        for s in range(7):
            col = m_idx * 7 + s
            
            if s == 0:
                # M1 = (a+d)(x+w)
                U[a, col] += 1
                U[d, col] += 1
                V[x, col] += 1
                V[w, col] += 1
                W[p00, col] += 1
                W[p11, col] += 1
            elif s == 1:
                # M2 = (c+d)x
                U[c, col] += 1
                U[d, col] += 1
                V[x, col] += 1
                W[p10, col] += 1
            elif s == 2:
                # M3 = a(y-w)
                U[a, col] += 1
                V[y, col] += 1
                V[w, col] -= 1
                W[p01, col] += 1
                W[p11, col] += 1
            elif s == 3:
                # M4 = d(z-x)
                U[d, col] += 1
                V[z, col] += 1
                V[x, col] -= 1
                W[p00, col] += 1
                W[p10, col] += 1
            elif s == 4:
                # M5 = (a+b)w
                U[a, col] += 1
                U[b, col] += 1
                V[w, col] += 1
                W[p00, col] -= 1
                W[p01, col] += 1
            elif s == 5:
                # M6 = (c-a)(x+y)
                U[c, col] += 1
                U[a, col] -= 1
                V[x, col] += 1
                V[y, col] += 1
                W[p11, col] += 1
            elif s == 6:
                # M7 = (b-d)(z+w)
                U[b, col] += 1
                U[d, col] -= 1
                V[z, col] += 1
                V[w, col] += 1
                W[p00, col] += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
