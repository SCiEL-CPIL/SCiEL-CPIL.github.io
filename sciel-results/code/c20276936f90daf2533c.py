import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # 2x2 Strassen coefficients
    # U_2x2: rows are A00, A01, A10, A11 (flattened i*2+j)
    # Cols are M0..M6
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    
    # V_2x2: rows are B00, B01, B10, B11 (flattened j*2+k)
    V_2x2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    
    # W_2x2: rows are C00, C01, C10, C11 (flattened i*2+k)
    W_2x2 = np.array([
        [1, 0, 0, 1, 0, -1, -1], # C00
        [0, 0, 0, 0, 1, 0, 1],   # C01
        [0, 0, 1, 0, 0, 1, 0],   # C10
        [1, -1, 1, 0, 1, 0, 0]   # C11
    ])
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U construction
            # Global index: (bi*2+ei)*4 + (bj*2+ej)
            # Coefficient: U_2x2[bi*2+bj, s] * U_2x2[ei*2+ej, t]
            for bi in range(2):
                for bj in range(2):
                    c_block = U_2x2[bi*2 + bj, s]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_elem = U_2x2[ei*2 + ej, t]
                            if c_elem == 0:
                                continue
                            row_a = (bi*2 + ei)*4 + (bj*2 + ej)
                            U[row_a, col] += c_block * c_elem
            
            # V construction
            # Global index: (bj*2+ej)*4 + (bk*2+ek)
            # Coefficient: V_2x2[bj*2+bk, s] * V_2x2[ej*2+ek, t]
            for bj in range(2):
                for bk in range(2):
                    c_block = V_2x2[bj*2 + bk, s]
                    if c_block == 0:
                        continue
                    for ej in range(2):
                        for ek in range(2):
                            c_elem = V_2x2[ej*2 + ek, t]
                            if c_elem == 0:
                                continue
                            row_b = (bj*2 + ej)*4 + (bk*2 + ek)
                            V[row_b, col] += c_block * c_elem
            
            # W construction
            # Global index: k*4 + i = (bk*2+ek)*4 + (bi*2+ei)
            # Coefficient: W_2x2[bi*2+bk, s] * W_2x2[ei*2+ek, t]
            for bi in range(2):
                for bk in range(2):
                    c_block = W_2x2[bi*2 + bk, s]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ek in range(2):
                            c_elem = W_2x2[ei*2 + ek, t]
                            if c_elem == 0:
                                continue
                            i = bi*2 + ei
                            k = bk*2 + ek
                            row_w = k*4 + i
                            W[row_w, col] += c_block * c_elem

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
