import numpy as np

DIMENSIONS = [4, 4, 9]

def _get_strassen_2x2_terms():
    """
    Returns lists U, V, W for 2x2 Strassen decomposition.
    Each list has 7 vectors of length 4.
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    U = []
    V = []
    W = []
    
    # M1 = (A11+A22)(B11+B22)
    u = np.array([1, 0, 0, 1], dtype=float)
    v = np.array([1, 0, 0, 1], dtype=float)
    w = np.array([1, 0, 0, 1], dtype=float) # C11 += 1, C22 += 1
    U.append(u); V.append(v); W.append(w)
    
    # M2 = (A21+A22)B11
    u = np.array([0, 0, 1, 1], dtype=float)
    v = np.array([1, 0, 0, 0], dtype=float)
    w = np.array([0, 0, 1, 0], dtype=float) # C21 += 1
    U.append(u); V.append(v); W.append(w)
    
    # M3 = A11(B12-B22)
    u = np.array([1, 0, 0, 0], dtype=float)
    v = np.array([0, 1, 0, -1], dtype=float)
    w = np.array([0, 1, 0, 0], dtype=float) # C12 += 1
    U.append(u); V.append(v); W.append(w)
    
    # M4 = A22(B21-B11)
    u = np.array([0, 0, 0, 1], dtype=float)
    v = np.array([0, 0, 1, -1], dtype=float)
    w = np.array([1, 0, 1, 0], dtype=float) # C11 += 1, C21 += 1
    U.append(u); V.append(v); W.append(w)
    
    # M5 = (A11+A12)B22
    u = np.array([1, 1, 0, 0], dtype=float)
    v = np.array([0, 0, 0, 1], dtype=float)
    w = np.array([-1, 1, 0, 0], dtype=float) # C11 -= 1, C12 += 1
    U.append(u); V.append(v); W.append(w)
    
    # M6 = (A21-A11)(B11+B12)
    u = np.array([0, 0, 1, -1], dtype=float)
    v = np.array([1, 1, 0, 0], dtype=float)
    w = np.array([0, 0, 1, 0], dtype=float) # C21 += 1
    U.append(u); V.append(v); W.append(w)
    
    # M7 = (A12-A22)(B21+B22)
    u = np.array([0, 1, 0, -1], dtype=float)
    v = np.array([0, 0, 1, 1], dtype=float)
    w = np.array([0, 0, 0, 1], dtype=float) # C22 += 1
    U.append(u); V.append(v); W.append(w)
    
    return U, V, W

def _build_4x4_strassen():
    """
    Builds the 49 rank-1 tensors for 4x4 x 4x4 multiplication using Strassen.
    Returns U, V, W lists of 49 vectors of length 16.
    U index: i*4+j
    V index: j*4+k
    W index: k*4+i
    """
    U2, V2, W2 = _get_strassen_2x2_terms()
    
    # Outer Strassen terms for 4x4
    # X_s is linear combo of A blocks (A00, A01, A10, A11)
    # Y_s is linear combo of B blocks (B00, B01, B10, B11)
    # W_s is linear combo of C blocks (C00, C01, C10, C11)
    
    # Coefficients for X_s (4 blocks)
    # Order: A00, A01, A10, A11
    X_coeffs = [
        [1, 0, 0, 1], # S1: A00 + A11
        [0, 0, 1, 1], # S2: A10 + A11
        [1, 0, 0, 0], # S3: A00
        [0, 0, 0, 1], # S4: A11
        [1, 1, 0, 0], # S5: A00 + A01
        [1, 0, -1, 0],# S6: A00 - A10
        [0, 1, 0, -1] # S7: A01 - A11
    ]
    
    # Coefficients for Y_s (4 blocks)
    # Order: B00, B01, B10, B11
    Y_coeffs = [
        [1, 0, 0, 1], # S1: B00 + B11
        [1, 0, 0, 0], # S2: B00
        [0, 1, 0, -1],# S3: B01 - B11
        [0, 0, 1, -1],# S4: B10 - B00
        [0, 0, 0, 1], # S5: B11
        [1, 1, 0, 0], # S6: B00 + B01
        [0, 0, 1, 1]  # S7: B10 + B11
    ]
    
    # Coefficients for W_s (4 blocks)
    # Order: C00, C01, C10, C11
    W_coeffs = [
        [1, 0, 0, 0], # S1 contributes to C00, C11? No, S1 is a 2x2 matrix.
        # Wait, S_s is a 2x2 matrix.
        # The result C is 4x4.
        # C00 = S1 + S4 - S5 - S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S2 + S3 + S7
        # So for each S_s, we add it to specific C blocks.
        # We need to distribute the 2x2 result of S_s into the 4x4 C.
        # S_s is a 2x2 matrix.
        # If S_s adds to C00, its elements map to C[0:2, 0:2].
        # If S_s adds to C01, its elements map to C[0:2, 2:4].
        # Etc.
        
        # We will handle this by iterating over the 7 inner terms of S_s.
        # For each inner term, we have a 2x2 vector w2 (length 4).
        # This w2 corresponds to the 2x2 result of the inner multiplication.
        # We need to map this 2x2 result to the appropriate 2x2 block of the 4x4 C.
        
        # So W_coeffs is not a single vector for the whole 4x4, but we need to know
        # which block of C the S_s result goes into.
        
        # Let's define the block destinations for S1..S7.
        # S1 -> C00 (+1), C11 (+1)
        # S2 -> C10 (+1), C11 (-1)
        # S3 -> C01 (+1), C11 (+1)
        # S4 -> C00 (+1), C10 (+1)
        # S5 -> C00 (-1), C01 (+1)
        # S6 -> C10 (+1)
        # S7 -> C00 (-1), C11 (+1)
    ]
    
    # Destinations: list of (block_row, block_col, coeff)
    # block_row, block_col in {0, 1}
    destinations = [
        [(0, 0, 1), (1, 1, 1)],       # S1
        [(1, 0, 1), (1, 1, -1)],      # S2
        [(0, 1, 1), (1, 1, 1)],       # S3
        [(0, 0, 1), (1, 0, 1)],       # S4
        [(0, 0, -1), (0, 1, 1)],      # S5
        [(1, 0, 1)],                  # S6
        [(0, 0, -1), (1, 1, 1)]       # S7
    ]
    
    U4 = []
    V4 = []
    W4 = []
    
    for s in range(7):
        for t in range(7):
            u2 = U2[t]
            v2 = V2[t]
            w2 = W2[t]
            
            # Construct U4 vector (length 16)
            # u2 is length 4 (2x2 flattened: i*2+j)
            # X_s is linear combo of A blocks.
            # A blocks are A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), etc.
            # A index: i*4 + j.
            # Block (br, bc) contains A[i, j] where i = br*2 + i_loc, j = bc*2 + j_loc.
            # Index in A: (br*2 + i_loc)*4 + (bc*2 + j_loc) = 8*br + 4*i_loc + 2*bc + j_loc.
            
            u4 = np.zeros(16, dtype=float)
            for i_loc in range(2):
                for j_loc in range(2):
                    val = u2[i_loc*2 + j_loc]
                    # Distribute to blocks based on X_coeffs[s]
                    for bc in range(2):
                        for br in range(2):
                            coeff = X_coeffs[s][br*2 + bc]
                            if coeff != 0:
                                idx = (br*2 + i_loc)*4 + (bc*2 + j_loc)
                                u4[idx] += coeff * val
                                
            # Construct V4 vector (length 16)
            # v2 is length 4 (2x2 flattened: j*2+k)
            # Y_s is linear combo of B blocks.
            # B blocks are B00, B01, B10, B11.
            # B index: j*4 + k.
            # Block (br, bc) contains B[j, k] where j = br*2 + j_loc, k = bc*2 + k_loc.
            # Index in B: (br*2 + j_loc)*4 + (bc*2 + k_loc) = 8*br + 4*j_loc + 2*bc + k_loc.
            
            v4 = np.zeros(16, dtype=float)
            for j_loc in range(2):
                for k_loc in range(2):
                    val = v2[j_loc*2 + k_loc]
                    for bc in range(2):
                        for br in range(2):
                            coeff = Y_coeffs[s][br*2 + bc]
                            if coeff != 0:
                                idx = (br*2 + j_loc)*4 + (bc*2 + k_loc)
                                v4[idx] += coeff * val
                                
            # Construct W4 vector (length 16)
            # w2 is length 4 (2x2 flattened: k*2+i)
            # Wait, the inner Strassen W index is k*2+i.
            # This corresponds to the 2x2 result matrix C_local.
            # C_local[k_loc, i_loc] = w2[k_loc*2 + i_loc].
            
            # We need to map C_local to the 4x4 C.
            # C index: k*4 + i.
            # If C_local goes to block (br, bc), then
            # C[k, i] where k = br*2 + k_loc, i = bc*2 + i_loc.
            # Index in C: (br*2 + k_loc)*4 + (bc*2 + i_loc) = 8*br + 4*k_loc + 2*bc + i_loc.
            
            w4 = np.zeros(16, dtype=float)
            for k_loc in range(2):
                for i_loc in range(2):
                    val = w2[k_loc*2 + i_loc]
                    for br, bc, coeff in destinations[s]:
                        if coeff != 0:
                            idx = (br*2 + k_loc)*4 + (bc*2 + i_loc)
                            w4[idx] += coeff * val
                            
            U4.append(u4)
            V4.append(v4)
            W4.append(w4)
            
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS # 4, 4, 9
    
    U_all = []
    V_all = []
    W_all = []
    
    # 4x4 x 4x9 = 4x4 x (4x4 + 4x4 + 4x1)
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    
    # Part 1: 4x4 x 4x4 (cols 0-3)
    U4, V4, W4 = _build_4x4_strassen()
    for i in range(len(U4)):
        # U4 is for 4x4 A, 4x4 B (cols 0-3)
        # U is same.
        # V needs to be embedded in 4x9 B.
        # V4 index j*4 + k_local.
        # Global V index j*9 + k_global.
        # k_global = k_local.
        u = np.zeros(16, dtype=float)
        v = np.zeros(36, dtype=float)
        w = np.zeros(36, dtype=float)
        
        u[:] = U4[i]
        
        for j in range(4):
            for k in range(4):
                v[j*9 + k] = V4[i][j*4 + k]
                
        # W4 index k*4 + i.
        # Global W index k*4 + i. (Since C is 4x9, index is k*4 + i? No.)
        # C is 4x9. W index is k*n + i = k*4 + i.
        # k is 0..8. i is 0..3.
        # For B1, k is 0..3.
        # So W4 index k*4 + i is the same as global W index k*4 + i.
        w[:] = W4[i]
        
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
        
    # Part 2: 4x4 x 4x4 (cols 4-7)
    for i in range(len(U4)):
        u = np.zeros(16, dtype=float)
        v = np.zeros(36, dtype=float)
        w = np.zeros(36, dtype=float)
        
        u[:] = U4[i]
        
        for j in range(4):
            for k in range(4):
                v[j*9 + (4 + k)] = V4[i][j*4 + k]
                
        for k in range(4):
            for i_idx in range(4):
                w[(4 + k)*4 + i_idx] = W4[i][k*4 + i_idx]
                
        U_all.append(u)
        V_all.append(v)
        W_all.append(w)
        
    # Part 3: 4x4 x 4x1 (col 8)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(1): # k=0
                u = np.zeros(16, dtype=float)
                v = np.zeros(36, dtype=float)
                w = np.zeros(36, dtype=float)
                
                u[i*4 + j] = 1.0
                v[j*9 + (8 + k)] = 1.0
                w[(8 + k)*4 + i] = 1.0
                
                U_all.append(u)
                V_all.append(v)
                W_all.append(w)
                
    rank = len(U_all)
    U = [[float(x) for x in row] for row in np.array(U_all).T]
    V = [[float(x) for x in row] for row in np.array(V_all).T]
    W = [[float(x) for x in row] for row in np.array(W_all).T]
    
    return {"U": U, "V": V, "W": W}
