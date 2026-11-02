import numpy as np
import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    
    # Rank 56
    # 8 blocks of 2x2x2 multiplication
    # Each block uses Strassen (rank 7)
    
    rank = 56
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    
    U = np.zeros((4, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Strassen decomposition for 2x2 matrix multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-c)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    
    # U coefficients for A elements [a, b, c, d]
    # Indices: 0:a, 1:b, 2:c, 3:d
    U_factors = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # V coefficients for B elements [e, f, g, h]
    # Indices: 0:e, 1:f, 2:g, 3:h
    V_factors = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coefficients for C elements [C11, C12, C21, C22]
    # Note: Task specifies W[k*n+i, r] where output is C[k, i]?
    # "Output factor follows the official trace convention W[k*n+i,r]"
    # C is 2x16. k is row (0..15? No, C is 2x16. Rows 0..1, Cols 0..15).
    # Wait, standard C = AB. A is 2x2, B is 2x16. C is 2x16.
    # Indices of C: i (0..1), j (0..15).
    # Convention W[k*n+i, r].
    # Usually k is the column index of the second matrix (B) and i is the row index of the first matrix (A)?
    # Or k is row of C, i is col of C?
    # "W[k*n+i, r]" suggests k goes from 0 to p-1? No, W has shape (p*n, rank).
    # p=16, n=2. p*n = 32.
    # If k is 0..15 and i is 0..1.
    # Index = k*2 + i.
    # This matches the shape 32.
    # So W index corresponds to C[k, i]? Or C[i, k]?
    # In matrix mult C = AB, C[i, j] = sum A[i, l] B[l, j].
    # Here A is 2x2, B is 2x16.
    # C[i, j] has i in 0..1, j in 0..15.
    # The prompt says W[k*n+i, r].
    # If n=2, index = 2k + i.
    # If k is the column index j of C (and B), and i is the row index i of C (and A).
    # Then index = 2*j + i.
    # This maps (j, i) to 2j+i.
    # So W[2j+i, r] corresponds to C[i, j].
    
    # Let's check the current program's W indexing.
    # Current:
    # idxs_w = [4*l, 4*l+1, 4*l+2, 4*l+3]
    # for l in 0..7.
    # Block l covers columns 2l and 2l+1 of B/C.
    # So j = 2l or 2l+1.
    # i = 0 or 1.
    # 
    # If index = 2*j + i:
    # j=2l, i=0 -> 2(2l) + 0 = 4l.
    # j=2l, i=1 -> 2(2l) + 1 = 4l+1.
    # j=2l+1, i=0 -> 2(2l+1) + 0 = 4l+2.
    # j=2l+1, i=1 -> 2(2l+1) + 1 = 4l+3.
    # 
    # So W[4l] is C[0, 2l]
    # W[4l+1] is C[1, 2l]
    # W[4l+2] is C[0, 2l+1]
    # W[4l+3] is C[1, 2l+1]
    
    # We need to assign Strassen W factors to these indices.
    # Strassen C = [[C11, C12], [C21, C22]]
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M3 + M6
    
    # For block l:
    # C[0, 2l] corresponds to C11.
    # C[1, 2l] corresponds to C21.
    # C[0, 2l+1] corresponds to C12.
    # C[1, 2l+1] corresponds to C22.
    
    # Indices in W for block l:
    # idx_c11 = 4l
    # idx_c21 = 4l+1
    # idx_c12 = 4l+2
    # idx_c22 = 4l+3
    
    # W coefficients for C11, C21, C12, C22 in terms of M1..M7
    # Order of M: 0..6
    W_factors = [
        [1, 0, 0, -1, -1, 0, 1], # C11: M1 - M4 - M5 + M7
        [0, 1, 0, 1, 0, 0, 0],   # C21: M2 + M4
        [0, 0, 1, 0, 1, 0, 0],   # C12: M3 + M5
        [1, -1, -1, 0, 0, 1, 0]  # C22: M1 - M2 - M3 + M6
    ]
    
    for l in range(8):
        for s in range(7):
            col = l * 7 + s
            
            # Fill U (same for all blocks)
            for i in range(4):
                U[i, col] = U_factors[s][i]
                
            # Fill V
            # B indices for block l:
            # B[0, 2l], B[0, 2l+1], B[1, 2l], B[1, 2l+1]
            # V shape is (m*p, rank) = (32, 56).
            # V index for B[r, c] is r*p + c = r*16 + c.
            
            # V_factors order: e, f, g, h
            # e = B[0, 2l] -> V[0*16 + 2l]
            # f = B[0, 2l+1] -> V[0*16 + 2l+1]
            # g = B[1, 2l] -> V[1*16 + 2l]
            # h = B[1, 2l+1] -> V[1*16 + 2l+1]
            
            v_indices = [
                0*16 + 2*l,
                0*16 + 2*l + 1,
                1*16 + 2*l,
                1*16 + 2*l + 1
            ]
            
            for idx, val in zip(v_indices, V_factors[s]):
                V[idx, col] = val
                
            # Fill W
            # W indices for block l:
            # C[0, 2l] -> W[4l]
            # C[1, 2l] -> W[4l+1]
            # C[0, 2l+1] -> W[4l+2]
            # C[1, 2l+1] -> W[4l+3]
            
            w_indices = [
                4*l,
                4*l + 1,
                4*l + 2,
                4*l + 3
            ]
            
            for idx, val in zip(w_indices, W_factors[s]):
                W[idx, col] = val
                
    # Convert to list of lists for return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
