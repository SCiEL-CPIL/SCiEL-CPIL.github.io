import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Part 1: A_L B_L
    # A_L is 2x2 (cols 0,1 of A)
    # B_L is 2x10 (rows 0,1 of B)
    # Split B_L into 5 blocks of 2x2
    
    for k in range(5):
        base_col = 7 * k
        # A indices: 0,1,3,4 (A00, A01, A10, A11)
        # B indices for block k: 
        # x = B[0, 2k] -> index 2k
        # y = B[0, 2k+1] -> index 2k+1
        # z = B[1, 2k] -> index 10 + 2k
        # w = B[1, 2k+1] -> index 10 + 2k + 1
        
        x_idx = 2 * k
        y_idx = 2 * k + 1
        z_idx = 10 + 2 * k
        w_idx = 10 + 2 * k + 1
        
        # T1: (a+d)(x+w)
        c1 = base_col
        U[0, c1] += 1  # a
        U[3, c1] += 1  # d
        V[x_idx, c1] += 1
        V[w_idx, c1] += 1
        
        # T2: (c+d)x
        c2 = base_col + 1
        U[3, c2] += 1  # c
        U[4, c2] += 1  # d
        V[x_idx, c2] += 1
        
        # T3: a(y-w)
        c3 = base_col + 2
        U[0, c3] += 1  # a
        V[y_idx, c3] += 1
        V[w_idx, c3] -= 1
        
        # T4: d(z-x)
        c4 = base_col + 3
        U[4, c4] += 1  # d
        V[z_idx, c4] += 1
        V[x_idx, c4] -= 1
        
        # T5: (a+b)w
        c5 = base_col + 4
        U[0, c5] += 1  # a
        U[1, c5] += 1  # b
        V[w_idx, c5] += 1
        
        # T6: (c-a)(x+z)
        c6 = base_col + 5
        U[3, c6] += 1  # c
        U[0, c6] -= 1  # a
        V[x_idx, c6] += 1
        V[z_idx, c6] += 1
        
        # T7: (b-d)(y+w)
        c7 = base_col + 6
        U[1, c7] += 1  # b
        U[4, c7] -= 1  # d
        V[y_idx, c7] += 1
        V[w_idx, c7] += 1
        
        # W coefficients
        # C[0, 2k] = T1 - T4 - T5 + T7
        W[2*k*2 + 0, c1] += 1
        W[2*k*2 + 0, c4] -= 1
        W[2*k*2 + 0, c5] -= 1
        W[2*k*2 + 0, c7] += 1
        
        # C[0, 2k+1] = T3 + T5
        W[(2*k+1)*2 + 0, c3] += 1
        W[(2*k+1)*2 + 0, c5] += 1
        
        # C[1, 2k] = T2 + T4
        W[2*k*2 + 1, c2] += 1
        W[2*k*2 + 1, c4] += 1
        
        # C[1, 2k+1] = T1 - T2 - T3 + T6
        W[(2*k+1)*2 + 1, c1] += 1
        W[(2*k+1)*2 + 1, c2] -= 1
        W[(2*k+1)*2 + 1, c3] -= 1
        W[(2*k+1)*2 + 1, c6] += 1

    # Part 2: A_R B_R
    # A_R is 2x1 (col 2 of A)
    # B_R is 1x10 (row 2 of B)
    
    for j in range(10):
        c1 = 35 + 2 * j
        c2 = 35 + 2 * j + 1
        
        # C[0, j] = A[0, 2] * B[2, j]
        U[2, c1] += 1
        V[20 + j, c1] += 1
        W[j*2 + 0, c1] += 1
        
        # C[1, j] = A[1, 2] * B[2, j]
        U[5, c2] += 1
        V[20 + j, c2] += 1
        W[j*2 + 1, c2] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
