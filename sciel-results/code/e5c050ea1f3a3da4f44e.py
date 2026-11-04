import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrix multiplication
    # U: (i,j) -> index i*2+j
    # V: (j,k) -> index j*2+k
    # W: (k,i) -> index k*2+i (Trace convention)
    
    # 7 columns for rank 7
    # Rows for U, V are 4 (2x2 inputs)
    # Rows for W are 4 (2x2 outputs)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a (00)
        [0, 0, 0, 0, 1, 0, 1],   # b (01)
        [0, 1, 0, 0, 0, 1, 0],   # c (10)
        [1, 1, 0, 1, 0, 0, -1]   # d (11)
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # e (00)
        [0, 0, 1, 0, 0, 1, 0],   # f (01)
        [0, 0, 0, 1, 0, 0, 1],   # g (10)
        [1, 0, -1, 0, 1, 0, 1]   # h (11)
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (k=0, i=1)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # C11 (k=1, i=1)
    ], dtype=int)
    
    # Construct 4x4 decomposition using Strassen block recursion
    # Rank 49
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    # Strassen block coefficients
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # A blocks: 0:A11, 1:A12, 2:A21, 3:A22
    # B blocks: 0:B11, 1:B12, 2:B21, 3:B22
    
    # Coefficients for A in each M_k (size 7x4)
    A_coeffs = np.array([
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 0, 1, 0],# M6: A21 - A11
        [0, 1, 0, -1] # M7: A12 - A22
    ], dtype=int)
    
    # Coefficients for B in each M_k (size 7x4)
    B_coeffs = np.array([
        [1, 0, 0, 1], # M1: B11 + B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12 - B22
        [0, 0, 1, -1],# M4: B21 - B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11 + B12
        [0, 0, 1, 1]  # M7: B21 + B22
    ], dtype=int)
    
    # Coefficients for C blocks in each M_k (size 4x7)
    # C11 = M1 - M4 + M7 - M2
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M5 + M6 - M3
    # C blocks: 0:C11, 1:C12, 2:C21, 3:C22
    C_coeffs = np.array([
        [1, -1, 0, 1, 0, 0, -1], # C11 from M1..M7
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, 0, -1, 0, -1, 1, 0]  # C22
    ], dtype=int)
    
    # Mapping of local 2x2 indices to global 4x4 indices
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global indices for C (trace convention k*4+i):
    
    # C11 (i in 0,1; k in 0,1)
    map_C11 = [0, 4, 1, 5]
    # C12 (i in 0,1; k in 2,3)
    map_C12 = [8, 12, 9, 13]
    # C21 (i in 2,3; k in 0,1)
    map_C21 = [2, 6, 3, 7]
    # C22 (i in 2,3; k in 2,3)
    map_C22 = [10, 14, 11, 15]
    
    # Mapping of A blocks to global indices
    # A11: 0,1,2,3
    # A12: 4,5,6,7
    # A21: 8,9,10,11
    # A22: 12,13,14,15
    map_A_blocks = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # Mapping of B blocks to global indices
    map_B_blocks = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    # Mapping of C blocks to global indices (for W)
    map_C_blocks = [
        map_C11,
        map_C12,
        map_C21,
        map_C22
    ]
    
    for k in range(7):
        for r in range(7):
            col_idx = k * 7 + r
            
            # Fill U4
            # U2[:, r] is the vector of weights for the 4 entries of the 2x2 input A
            u_vec = U2[:, r]
            for b_idx in range(4):
                coeff = A_coeffs[k, b_idx]
                if coeff != 0:
                    global_indices = map_A_blocks[b_idx]
                    for l_idx in range(4):
                        U4[global_indices[l_idx], col_idx] += coeff * u_vec[l_idx]
                        
            # Fill V4
            v_vec = V2[:, r]
            for b_idx in range(4):
                coeff = B_coeffs[k, b_idx]
                if coeff != 0:
                    global_indices = map_B_blocks[b_idx]
                    for l_idx in range(4):
                        V4[global_indices[l_idx], col_idx] += coeff * v_vec[l_idx]
                        
            # Fill W4
            w_vec = W2[:, r]
            for c_idx in range(4):
                coeff = C_coeffs[c_idx, k]
                if coeff != 0:
                    global_indices = map_C_blocks[c_idx]
                    for l_idx in range(4):
                        W4[global_indices[l_idx], col_idx] += coeff * w_vec[l_idx]
                        
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
