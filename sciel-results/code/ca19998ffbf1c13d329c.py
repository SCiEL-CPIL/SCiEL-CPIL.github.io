import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    # Rank is 98
    
    # 1. Construct 2x2 rank-7 decomposition
    # U22, V22, W22 are lists of 7 vectors of length 4
    U22 = []
    V22 = []
    W22 = []
    
    # p1 = (a+d)(e+h) -> w: c00+=1, c11+=1
    U22.append([1, 0, 0, 1])
    V22.append([1, 0, 0, 1])
    W22.append([1, 0, 0, 1]) # c00, c01, c10, c11
    
    # p2 = (c+d)e -> w: c10+=1, c11-=1
    U22.append([0, 0, 1, 1])
    V22.append([1, 0, 0, 0])
    W22.append([0, 0, 1, -1])
    
    # p3 = a(f-h) -> w: c01+=1, c11+=1
    U22.append([1, 0, 0, 0])
    V22.append([0, 1, 0, -1])
    W22.append([0, 1, 0, 1])
    
    # p4 = d(g-e) -> w: c00+=1, c10+=1
    U22.append([0, 0, 0, 1])
    V22.append([0, 0, 1, -1])
    W22.append([1, 0, 1, 0])
    
    # p5 = (a+b)h -> w: c00-=1, c01+=1
    U22.append([1, 1, 0, 0])
    V22.append([0, 0, 0, 1])
    W22.append([-1, 1, 0, 0])
    
    # p6 = (c-a)(e+f) -> w: c11+=1
    U22.append([-1, 0, 1, 0])
    V22.append([1, 1, 0, 0])
    W22.append([0, 0, 0, 1])
    
    # p7 = (b-d)(g+h) -> w: c00+=1
    U22.append([0, 1, 0, -1])
    V22.append([0, 0, 1, 1])
    W22.append([1, 0, 0, 0])
    
    # 2. Construct 4x4 rank-49 decomposition
    # U44, V44, W44 will be lists of 49 vectors of length 16
    U44 = []
    V44 = []
    W44 = []
    
    # Block indices in 4x4 matrix (row-major)
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # Helper to expand a 2x2 block combination to 4x4 vector
    def expand_block(vec2x2, block_indices):
        # vec2x2 is [v00, v01, v10, v11]
        # block_indices is list of 4 indices in 16-dim space
        res = [0]*16
        for i in range(4):
            res[block_indices[i]] += vec2x2[i]
        return res
    
    # Block index maps
    # A blocks
    A00_idx = [0, 1, 4, 5]
    A01_idx = [2, 3, 6, 7]
    A10_idx = [8, 9, 12, 13]
    A11_idx = [10, 11, 14, 15]
    
    # B blocks (same indices for 4x4)
    B00_idx = A00_idx
    B01_idx = A01_idx
    B10_idx = A10_idx
    B11_idx = A11_idx
    
    # C blocks (same indices for 4x4)
    C00_idx = A00_idx
    C01_idx = A01_idx
    C10_idx = A10_idx
    C11_idx = A11_idx
    
    # Strassen steps for 4x4
    # Each step: (X_comb, Y_comb, C_comb)
    # X_comb: dict of block_name -> coeff
    # Y_comb: dict of block_name -> coeff
    # C_comb: dict of block_name -> coeff
    
    steps = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1}), # M1
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1, 'C11': -1}),          # M2
        ({'A00': 1}, {'B01': 1, 'B11': -1}, {'C01': 1, 'C11': 1}),           # M3
        ({'A11': 1}, {'B10': 1, 'B00': -1}, {'C00': 1, 'C10': 1}),           # M4
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': -1, 'C01': 1}),           # M5
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B01': 1}, {'C11': 1}),           # M6
        ({'A01': 1, 'A11': -1}, {'B10': 1, 'B11': 1}, {'C00': 1})            # M7
    ]
    
    block_map_A = {'A00': A00_idx, 'A01': A01_idx, 'A10': A10_idx, 'A11': A11_idx}
    block_map_B = {'B00': B00_idx, 'B01': B01_idx, 'B10': B10_idx, 'B11': B11_idx}
    block_map_C = {'C00': C00_idx, 'C01': C01_idx, 'C10': C10_idx, 'C11': C11_idx}
    
    for x_comb, y_comb, c_comb in steps:
        # Construct X and Y as 4x4 vectors (length 16)
        X = [0]*16
        Y = [0]*16
        for block, coeff in x_comb.items():
            idxs = block_map_A[block]
            for i in range(4):
                X[idxs[i]] += coeff
        for block, coeff in y_comb.items():
            idxs = block_map_B[block]
            for i in range(4):
                Y[idxs[i]] += coeff
        
        # For each of the 7 rank-1 components of 2x2 mult
        for r in range(7):
            u2 = U22[r]
            v2 = V22[r]
            w2 = W22[r]
            
            # Expand u2, v2 to length 16 using X, Y?
            # No. u2 is a vector of coefficients for the entries of X.
            # X is a linear combination of A.
            # The product is (u2 dot X) * (v2 dot Y)?
            # No. u2, v2 are the vectors in the 2x2 decomposition.
            # The term is w2 (output vector) * v2 (B vector) * u2 (A vector).
            # But u2 and v2 are defined on the 2x2 blocks.
            # We need to map u2 to the 4x4 A space.
            # u2[0] corresponds to X00, u2[1] to X01, etc.
            # X00 is a linear combination of A.
            # So the coefficient for A_ij in the final U vector is sum over blocks.
            
            # Let's compute the full U, V, W vectors for this component.
            U_full = [0]*16
            V_full = [0]*16
            W_full = [0]*16
            
            # U_full: coefficients for A
            # u2[k] is coeff for X_k.
            # X_k is sum of A_blocks.
            # We already built X as a vector of coefficients for A.
            # Wait. X is the vector of coefficients.
            # The "input" to the 2x2 mult is the matrix X.
            # The rank-1 term uses u2 dot X_matrix.
            # u2 is a vector. X_matrix is a matrix.
            # u2 dot X_matrix = sum_k u2[k] * X_matrix[k].
            # X_matrix[k] is a linear combination of A.
            # So U_full = sum_k u2[k] * (vector for X_k).
            # We have X as a vector of length 16, which is sum_k 1 * (vector for X_k) if X_k were 1?
            # No. X is the specific combination for this Strassen step.
            # X is a vector of length 16.
            # But u2 is a vector of length 4.
            # We need to form the vector $U_{term} = \sum_{k=0}^3 u2[k] \cdot X_k$.
            # But $X$ is $\sum_{blocks} c_b \cdot Block_b$.
            # This is getting confusing.
            
            # Let's restart the expansion logic.
            # X is a 2x2 matrix of linear forms in A.
            # X = c00*A00 + c01*A01 + ...
            # Actually, in my `steps`, X is a sum of blocks.
            # e.g. X = A00 + A11.
            # So X00 = A00_00 + A11_00.
            # X01 = A00_01 + A11_01.
            # etc.
            
            # u2 is [u00, u01, u10, u11].
            # The scalar product is u00*X00 + u01*X01 + u10*X10 + u11*X11.
            # Each X_ij is a linear combination of A.
            # So U_full is the sum of u_ij * (vector for X_ij).
            
            # I need the vectors for X00, X01, X10, X11.
            # From x_comb:
            # X00 vector: sum of coeff * A_block_vec for blocks in x_comb.
            # But A_block_vec is a vector of 4 elements (the block itself).
            # And X00 is the top-left element of the 2x2 matrix X.
            # So X00 = sum_{block} coeff_block * (block[0]).
            # block[0] is the top-left element of the block in A.
            
            # Let's precompute the vectors for the 4 elements of X and Y.
            X_vecs = []
            Y_vecs = []
            for k in range(4): # 0,1,2,3 correspond to 00,01,10,11
                xv = [0]*16
                yv = [0]*16
                for block, coeff in x_comb.items():
                    idxs = block_map_A[block]
                    xv[idxs[k]] += coeff
                for block, coeff in y_comb.items():
                    idxs = block_map_B[block]
                    yv[idxs[k]] += coeff
                X_vecs.append(xv)
                Y_vecs.append(yv)
                
            # Now compute U_full, V_full
            for k in range(4):
                U_full = [u + u2[k]*xv for u, xv in zip(U_full, X_vecs[k])]
                V_full = [v + v2[k]*yv for v, yv in zip(V_full, Y_vecs[k])]
                
            # W_full
            # w2 is [w00, w01, w10, w11] for the 2x2 output M.
            # M adds to C blocks according to c_comb.
            # M00 adds to C_blocks with coeff.
            # W_full = sum_k w2[k] * (vector for M_k contribution to C).
            for k in range(4):
                wv = [0]*16
                for block, coeff in c_comb.items():
                    idxs = block_map_C[block]
                    wv[idxs[k]] += coeff
                W_full = [w + w2[k]*wv for w, wv in zip(W_full, wv)]
                
            U44.append(U_full)
            V44.append(V_full)
            W44.append(W_full)
            
    # 3. Construct 4x4x8 rank-98 decomposition
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # Left half (cols 0-3)
    for r in range(49):
        for i in range(16):
            U[i][r] = U44[r][i]
            # V index: j*8 + k. Here k in 0..3.
            # U44 index is j*4 + k.
            # So V[j*8 + k] = V44[j*4 + k].
            j = i // 4
            k = i % 4
            V[j*8 + k][r] = V44[r][i]
            
            # W index: k*4 + i. Here k in 0..3.
            # W44 index is k*4 + i.
            # So W[k*4 + i] = W44[k*4 + i].
            # Note: W44 is built with C indices row-major i*4+k?
            # My W44 construction used C00, C01, C10, C11 blocks.
            # C00 is rows 0-1, cols 0-1.
            # C index in W44 is i*4+k?
            # Let's check W44 indices.
            # I used C00_idx = [0,1,4,5]. This is row-major.
            # So W44[r][i] corresponds to C[i//4, i%4].
            # The task requires W[k*n+i].
            # So I need to transpose the W44 indices.
            # W_task[k*4+i] = W44[i*4+k].
            i_row = i // 4
            k_col = i % 4
            # W_task index is k_col*4 + i_row.
            W[k_col*4 + i_row][r] = W44[r][i]
            
    # Right half (cols 4-7)
    for r in range(49):
        for i in range(16):
            U[i][49+r] = U44[r][i]
            # V index: j*8 + (k+4).
            j = i // 4
            k = i % 4
            V[j*8 + (k+4)][49+r] = V44[r][i]
            
            # W index: (k+4)*4 + i.
            i_row = i // 4
            k_col = i % 4
            W[(k_col+4)*4 + i_row][49+r] = W44[r][i]
            
    return {"U": U, "V": V, "W": W}
