import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Strassen blocks for A[:, :2] * B[:2, :]
    # Split B[:2, :] into 3 blocks of width 2 and 1 block of width 1
    # Blocks: k in [0,2), [2,4), [4,6), [6,7)
    
    strassen_u = [
        [1, 0, 0, 0], # M1: A00
        [1, 0, 1, 0], # M2: A00 + A10
        [0, 0, 1, 1], # M3: A10 + A11
        [0, 0, 0, 1], # M4: A11
        [1, 0, 0, 1], # M5: A00 + A11
        [0, 1, 0, -1],# M6: A01 - A11
        [1, 0, -1, 0] # M7: A00 - A10
    ]
    # Indices in strassen_u: [A00, A01, A10, A11] -> j=0,1; i=0,1
    # Actually, let's map carefully.
    # A is 2x2. A[i,j].
    # M1 = A00 * (B01 + B11)
    # M2 = (A00 + A10) * B10
    # M3 = (A10 + A11) * (B00 - B01)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A00 - A10) * (B00 + B01)
    
    # Let's define coefficients for U (A part) and V (B part) for each of the 7 products
    # U coeffs: for i in 0,1 and j in 0,1
    # V coeffs: for j in 0,1 and k in 0,1 (local)
    
    u_coeffs = [
        [[1, 0], [0, 0]], # M1: A00
        [[1, 0], [1, 0]], # M2: A00 + A10
        [[0, 0], [1, 1]], # M3: A10 + A11
        [[0, 0], [0, 1]], # M4: A11
        [[1, 0], [0, 1]], # M5: A00 + A11
        [[0, 1], [0, -1]],# M6: A01 - A11
        [[1, 0], [-1, 0]] # M7: A00 - A10
    ]
    
    v_coeffs = [
        [[0, 1], [0, 1]], # M1: B01 + B11
        [[0, 0], [1, 0]], # M2: B10
        [[1, -1], [0, 0]],# M3: B00 - B01
        [[-1, 0], [1, 0]],# M4: B10 - B00
        [[1, 0], [0, 1]], # M5: B00 + B11
        [[1, 1], [0, 1]], # M6: B10 + B11
        [[1, 1], [0, 0]]  # M7: B00 + B01
    ]
    
    w_coeffs = [
        [[0, 0], [0, 0]], # placeholder, filled below
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]]
    ]
    
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M7
    # C10 = M3 + M5
    # C11 = M1 - M4 + M2 + M7
    
    # w_coeffs[s][i][k]
    w_coeffs[0] = [[0, 1], [0, 0]] # M1 -> C01
    w_coeffs[1] = [[0, 0], [0, 1]] # M2 -> C11
    w_coeffs[2] = [[0, 0], [1, 0]] # M3 -> C10
    w_coeffs[3] = [[1, 0], [0, -1]] # M4 -> C00, C11
    w_coeffs[4] = [[1, 0], [1, 0]] # M5 -> C00, C10
    w_coeffs[5] = [[1, 0], [0, 0]] # M6 -> C00
    w_coeffs[6] = [[0, 1], [0, 1]] # M7 -> C01, C11
    
    # Check C00: M5(1) + M4(1) - M2(0? No, M2 is not in C00 formula? Wait)
    # Formula: C00 = M5 + M4 - M2 + M6
    # My w_coeffs:
    # M5: C00=1
    # M4: C00=1
    # M2: C00=0? I put M2 -> C11.
    # Let's re-verify Strassen formulas.
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M7
    # C10 = M3 + M5
    # C11 = M1 - M4 + M2 + M7
    
    # So:
    # M1: C01=1, C11=1
    # M2: C00=-1, C11=1
    # M3: C10=1
    # M4: C00=1, C11=-1
    # M5: C00=1, C10=1
    # M6: C00=1
    # M7: C01=1, C11=1
    
    w_coeffs = [
        [[0, 1], [0, 0]], # M1
        [[-1, 0], [0, 1]], # M2
        [[0, 0], [1, 0]], # M3
        [[1, 0], [0, -1]], # M4
        [[1, 0], [1, 0]], # M5
        [[1, 0], [0, 0]], # M6
        [[0, 1], [0, 1]]  # M7
    ]
    
    # Now populate U, V, W for the 3 Strassen blocks
    for c in range(3):
        for s in range(7):
            r = col
            col += 1
            # U
            for i in range(2):
                for j in range(2):
                    val = u_coeffs[s][i][j]
                    if val != 0:
                        U[i * 3 + j, r] += val
            # V
            for j in range(2):
                for k_loc in range(2):
                    k_glob = 2 * c + k_loc
                    val = v_coeffs[s][j][k_loc]
                    if val != 0:
                        V[j * 7 + k_glob, r] += val
            # W
            for i in range(2):
                for k_loc in range(2):
                    k_glob = 2 * c + k_loc
                    val = w_coeffs[s][i][k_loc]
                    if val != 0:
                        W[k_glob * 2 + i, r] += val
                        
    # Schoolbook for A[:, :2] * B[:2, 6:7]
    # A is 2x2 (cols 0,1), B is 2x1 (col 6)
    # C is 2x1 (col 6)
    # Terms: (i, j) for i in 0,1, j in 0,1
    for i in range(2):
        for j in range(2):
            r = col
            col += 1
            U[i * 3 + j, r] = 1
            V[j * 7 + 6, r] = 1
            W[6 * 2 + i, r] = 1
            
    # Schoolbook for A[:, 2:3] * B[2:3, :]
    # A is 2x1 (col 2), B is 1x7 (row 2)
    # C is 2x7
    # Terms: (i, k) for i in 0,1, k in 0..6
    for i in range(2):
        for k in range(7):
            r = col
            col += 1
            U[i * 3 + 2, r] = 1
            V[2 * 7 + k, r] = 1
            W[k * 2 + i, r] = 1
            
    # Convert to lists of lists or keep as numpy? 
    # The task says "factor matrices of shapes ... with integer/half-integer entries"
    # Returning numpy arrays is usually fine if they are integer types, but lists are safer for "integer entries" if the checker is strict about types.
    # However, numpy int arrays are fine.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
