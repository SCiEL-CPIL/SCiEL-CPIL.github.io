import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define 2x2 Strassen decomposition (rank 7)
    # U2, V2, W2 are 4x7 matrices
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # t=0: M1 = (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1: M2 = (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # t=2: M3 = a(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # t=3: M4 = d(g-e)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # t=4: M5 = (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # t=5: M6 = (c-a)(e+f)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6: M7 = (b-d)(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Define 3x3 Schoolbook decomposition (rank 27)
    # U3, V3, W3 are 9x27 matrices
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                t3 = i3 * 9 + j3 * 3 + k3
                r_u3 = i3 * 3 + j3
                r_v3 = j3 * 3 + k3
                r_w3 = k3 * 3 + i3
                U3[r_u3, t3] = 1
                V3[r_v3, t3] = 1
                W3[r_w3, t3] = 1

    # Compute Kronecker products
    # U_kron shape: (9*4, 27*7) = (36, 189)
    # Row index in kron: r3 * 4 + r2
    # We need to permute rows to match i*6+j ordering where i = 2*i3+i2, j = 2*j3+j2
    # r_u3 = i3*3 + j3, r_u2 = i2*2 + j2
    # kron_row = r_u3 * 4 + r_u2
    # target_row = i * 6 + j = (2*i3+i2)*6 + (2*j3+j2)
    
    U_kron = np.kron(U3, U2)
    V_kron = np.kron(V3, V2)
    W_kron = np.kron(W3, W2)
    
    rank = 189
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    # Build permutation mapping for rows
    # For each target row index (0..35), find the corresponding kron row index
    # target_row = i*6+j
    # i = 2*i3 + i2, j = 2*j3 + j2
    # r_u3 = i3*3 + j3, r_u2 = i2*2 + j2
    # kron_row = r_u3 * 4 + r_u2
    
    # We'll fill U, V, W by iterating over target rows
    for i in range(6):
        for j in range(6):
            target_row = i * 6 + j
            i3 = i // 2
            i2 = i % 2
            j3 = j // 2
            j2 = j % 2
            
            r_u3 = i3 * 3 + j3
            r_u2 = i2 * 2 + j2
            kron_row_u = r_u3 * 4 + r_u2
            
            r_v3 = j3 * 3 + j2  # Wait, V index is j,k. Let me recheck.
            # V rows are indexed by (j, k) in the 6x6 case.
            # In the kron structure:
            # V3 rows are (j3, k3) -> r_v3 = j3*3 + k3
            # V2 rows are (j2, k2) -> r_v2 = j2*2 + k2
            # So for V, the row depends on j and k, not i and j.
            
    # Let's do this more carefully by iterating over all indices for each matrix separately.
    
    # For U (rows are i,j):
    for i in range(6):
        for j in range(6):
            target_row = i * 6 + j
            i3 = i // 2
            i2 = i % 2
            j3 = j // 2
            j2 = j % 2
            
            r_u3 = i3 * 3 + j3
            r_u2 = i2 * 2 + j2
            kron_row = r_u3 * 4 + r_u2
            
            U[target_row, :] = U_kron[kron_row, :]
    
    # For V (rows are j,k):
    for j in range(6):
        for k in range(6):
            target_row = j * 6 + k
            j3 = j // 2
            j2 = j % 2
            k3 = k // 2
            k2 = k % 2
            
            r_v3 = j3 * 3 + k3
            r_v2 = j2 * 2 + k2
            kron_row = r_v3 * 4 + r_v2
            
            V[target_row, :] = V_kron[kron_row, :]
    
    # For W (rows are k,i):
    for k in range(6):
        for i in range(6):
            target_row = k * 6 + i
            k3 = k // 2
            k2 = k % 2
            i3 = i // 2
            i2 = i % 2
            
            r_w3 = k3 * 3 + i3
            r_w2 = k2 * 2 + i2
            kron_row = r_w3 * 4 + r_w2
            
            W[target_row, :] = W_kron[kron_row, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
