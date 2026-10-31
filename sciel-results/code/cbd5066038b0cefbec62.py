import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 matrices
    # S_A, S_B, S_C are 7x2x2 arrays
    # Indices: [r, i, j]
    
    S_A = np.zeros((7, 2, 2), dtype=int)
    S_B = np.zeros((7, 2, 2), dtype=int)
    S_C = np.zeros((7, 2, 2), dtype=int)
    
    # r=0
    S_A[0, 0, 0] = 1; S_A[0, 1, 1] = 1
    S_B[0, 0, 0] = 1; S_B[0, 1, 1] = 1
    S_C[0, 0, 0] = 1; S_C[0, 1, 1] = 1
    
    # r=1
    S_A[1, 1, 0] = 1; S_A[1, 1, 1] = 1
    S_B[1, 0, 0] = 1
    S_C[1, 1, 0] = 1; S_C[1, 1, 1] = -1
    
    # r=2
    S_A[2, 0, 0] = 1
    S_B[2, 0, 1] = 1; S_B[2, 1, 1] = -1
    S_C[2, 0, 1] = 1; S_C[2, 1, 1] = 1
    
    # r=3
    S_A[3, 1, 1] = 1
    S_B[3, 1, 0] = 1; S_B[3, 0, 0] = -1
    S_C[3, 0, 0] = 1; S_C[3, 1, 0] = 1
    
    # r=4
    S_A[4, 0, 0] = 1; S_A[4, 0, 1] = 1
    S_B[4, 1, 1] = 1
    S_C[4, 0, 0] = -1; S_C[4, 0, 1] = 1
    
    # r=5
    S_A[5, 1, 0] = 1; S_A[5, 0, 0] = -1
    S_B[5, 0, 0] = 1; S_B[5, 0, 1] = 1
    S_C[5, 1, 1] = 1
    
    # r=6
    S_A[6, 0, 1] = 1; S_A[6, 1, 1] = -1
    S_B[6, 1, 0] = 1; S_B[6, 1, 1] = 1
    S_C[6, 0, 0] = 1
    
    # Construct U, V, W
    # The decomposition is tensor product of two Strassen decompositions.
    # Rank is 7 * 7 = 49.
    # Column index c = r1 * 7 + r2
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            # U contributions
            # U[global_row, col]
            # global_row corresponds to A[i, j] where i, j in 0..3
            # A is split into 2x2 blocks of 2x2.
            # Block indices I, J in 0..1. Local indices i, j in 0..1.
            # Global row index: (2*I + i) * 4 + (2*J + j)
            
            for I in range(2):
                for J in range(2):
                    a1 = S_A[r1, I, J]
                    if a1 == 0: continue
                    for i in range(2):
                        for j in range(2):
                            a2 = S_A[r2, i, j]
                            if a2 == 0: continue
                            val = a1 * a2
                            if val == 0: continue
                            row_idx = (2 * I + i) * 4 + (2 * J + j)
                            U[row_idx, col] += val
            
            # V contributions
            # V[global_row, col]
            # global_row corresponds to B[k, l] where k, l in 0..3
            # Block indices K, L in 0..1. Local indices k, l in 0..1.
            # Global row index: (2*K + k) * 4 + (2*L + l)
            
            for K in range(2):
                for L in range(2):
                    b1 = S_B[r1, K, L]
                    if b1 == 0: continue
                    for k in range(2):
                        for l in range(2):
                            b2 = S_B[r2, k, l]
                            if b2 == 0: continue
                            val = b1 * b2
                            if val == 0: continue
                            row_idx = (2 * K + k) * 4 + (2 * L + l)
                            V[row_idx, col] += val
            
            # W contributions
            # W[global_row, col]
            # global_row corresponds to C[row, col] -> W index is col * n + row ?
            # Task says: W[k*n+i, r].
            # Standard tensor T[i,j,k] = sum U[i,r] V[j,r] W[k,r].
            # Here A is n x m (4x4), B is m x p (4x4), C is n x p (4x4).
            # T[i, j, k] where i is row of A, j is col of A / row of B, k is col of B / col of C.
            # Wait, standard matrix mult: C = A B.
            # C[i, k] = sum_j A[i, j] B[j, k].
            # Tensor T[i, j, k] = delta.
            # U[i, r] corresponds to A[i, j]? No, U is (n*m, rank).
            # So U index is (i, j) flattened.
            # V index is (j, k) flattened.
            # W index is (i, k) flattened?
            # Task says W is (p*n, rank).
            # And "output factor follows the official trace convention W[k*n+i,r]".
            # This implies W row index is k*n + i.
            # Here k is the column of C (0..3), i is the row of C (0..3).
            # So W row index = k * 4 + i.
            
            for I in range(2):
                for J in range(2):
                    c1 = S_C[r1, I, J]
                    if c1 == 0: continue
                    for i in range(2):
                        for j in range(2):
                            c2 = S_C[r2, i, j]
                            if c2 == 0: continue
                            val = c1 * c2
                            if val == 0: continue
                            # C row is 2*I + i. C col is 2*J + j.
                            # W row index = C_col * 4 + C_row = (2*J + j) * 4 + (2*I + i).
                            row_idx = (2 * J + j) * 4 + (2 * I + i)
                            W[row_idx, col] += val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
