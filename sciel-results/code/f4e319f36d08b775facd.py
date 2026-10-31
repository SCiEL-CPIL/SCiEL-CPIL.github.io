import numpy as np

def construct():
    n, m, p = 4, 4, 4
    
    # Strassen factors for 2x2
    # U_S: 4x7, rows (i, j) 0-3
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=float)
    
    # V_S: 4x7, rows (j, k) 0-3
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [1, 0, 1, 0, 0, 1, 0],  # B01
        [1, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=float)
    
    # W_S: 4x7, rows (k, i) 0-3 -> k*2+i
    # C00 (k=0,i=0): M1 + M4 - M5 + M7
    # C10 (k=0,i=1): M2 + M4
    # C01 (k=1,i=0): M3 + M5
    # C11 (k=1,i=1): M1 - M2 + M3 + M6
    
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ], dtype=float)
    
    # Kronecker product
    # U_4: 16x49
    U_4 = np.kron(U_S, U_S)
    V_4 = np.kron(V_S, V_S)
    W_4 = np.kron(W_S, W_S)
    
    # Permute rows
    # U_4 rows are (i1, j1, i2, j2) in kron order?
    # np.kron(A, B) row index r = r_A * rows_B + r_B.
    # U_S rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U_4 row r corresponds to (r_A, r_B).
    # r_A encodes (i1, j1). r_B encodes (i2, j2).
    # We want row index i*m + j = i*4 + j.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # r_A = i1*2 + j1. r_B = i2*2 + j2.
    # r = r_A * 4 + r_B = (i1*2 + j1)*4 + (i2*2 + j2) = 8*i1 + 4*j1 + 2*i2 + j2.
    # Target index = (2*i1 + i2)*4 + (2*j1 + j2) = 8*i1 + 4*i2 + 2*j1 + j2.
    # So we need to permute rows of U_4.
    
    def permute_rows(mat, n):
        # n=4
        # r = 8*i1 + 4*j1 + 2*i2 + j2
        # target = 8*i1 + 4*i2 + 2*j1 + j2
        # Map r -> target
        perm = [0] * (n*n)
        for i1 in range(2):
            for i2 in range(2):
                i = 2*i1 + i2
                for j1 in range(2):
                    for j2 in range(2):
                        j = 2*j1 + j2
                        r = (i1*2 + j1)*4 + (i2*2 + j2)
                        target = i*4 + j
                        perm[r] = target
        # perm[r] is the target index for source row r
        # We want mat_new[target] = mat[r]
        # So mat_new = mat[inv_perm]
        inv_perm = [0] * (n*n)
        for r in range(n*n):
            inv_perm[perm[r]] = r
        return mat[inv_perm]
        
    U_final = permute_rows(U_4, 4)
    V_final = permute_rows(V_4, 4)
    W_final = permute_rows(W_4, 4)
    
    # Convert to lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
