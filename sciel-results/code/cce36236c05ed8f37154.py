import numpy as np

def construct():
    # 2x2 Strassen factors
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 0, 0, 1, 0],  # C10
        [0, 0, -1, 0, 1, 0, 0], # C01
        [1, 0, 1, 1, 0, 1, 0]   # C11
    ], dtype=int)
    
    # 4x4 factors
    # Shape (16, 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    for c1 in range(7):
                        for c2 in range(7):
                            c = 7 * c1 + c2
                            val_u = U2[i1, c1] * U2[i2, c2]
                            if val_u != 0:
                                U4[i * 4 + j, c] = val_u
                            
                            # V index: j, k
                            # j = 2*j1 + j2
                            # k = 2*k1 + k2
                            # We need to iterate k1, k2 for V
                            # But V4 depends on j and k.
                            # The loop above fixes j. We need k.
                            # It's better to loop over all indices independently.
    
    # Let's restructure the loop for efficiency and correctness
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for c1 in range(7):
        for c2 in range(7):
            c = 7 * c1 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = 2 * j1 + j2
                            # U
                            U4[i * 4 + j, c] = U2[i1, c1] * U2[i2, c2]
                            
                            # V
                            # V index is j*p + k. p=4.
                            # j = 2*j1 + j2
                            # k = 2*k1 + k2
                            # We need to fill V4 for all k.
                            # So we should loop k1, k2 inside or outside?
                            # V4[j*4+k, c] = V2[j1, c1] * V2[j2, c2] is wrong.
                            # V2 index is j, k.
                            # V2[j1, k1] * V2[j2, k2].
                            # So for a fixed j1, j2, c1, c2, we need to loop k1, k2.
                            pass

    # Correct construction:
    for c1 in range(7):
        for c2 in range(7):
            c = 7 * c1 + c2
            
            # U
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = 2 * j1 + j2
                            U4[i * 4 + j, c] = U2[i1, c1] * U2[i2, c2]
                            
            # V
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    for k1 in range(2):
                        for k2 in range(2):
                            k = 2 * k1 + k2
                            V4[j * 4 + k, c] = V2[j1, c1] * V2[j2, c2]
                            
            # W
            # W index is k*n + i. n=4.
            # k = 2*k1 + k2
            # i = 2*i1 + i2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    for i1 in range(2):
                        for i2 in range(2):
                            i = 2 * i1 + i2
                            W4[k * 4 + i, c] = W2[k1, c1] * W2[k2, c2]
                            
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
