import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 8
    
    # Base Strassen 2x2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 decomposition (Rank 49)
    # Indices for 4x4x4:
    # U: (i*4+j), V: (j*4+k), W: (k*4+i)
    # Decomposition: sum_{t,s} U2[i1*2+j1, t] U2[i2*2+j2, s] * V2[j1*2+k1, t] V2[j2*2+k2, s] * W2[k1*2+i1, t] W2[k2*2+i2, s]
    # where i = 2*i1+i2, j = 2*j1+j2, k = 2*k1+k2
    
    # We can form U4, V4, W4 by iterating t, s in 0..6
    # U4[:, t*7+s] is a vector of length 16.
    # U4[i*4+j, t*7+s] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for t in range(7):
        for s in range(7):
            c = t*7 + s
            # Construct the vector for this term
            # u_term[i*4+j] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
            # This is the Kronecker product of U2[:, t] and U2[:, s]
            # But careful with index ordering.
            # U2 is 4x7. U2[:, t] is length 4 (indices 0..3 correspond to i1*2+j1? No, U2 rows are i1*2+j1? 
            # In the base 2x2x2, U2 row index is i*2+j? No, for 2x2, i,j in 0..1.
            # U2 shape is (4, 7). Row index r = i*2 + j? 
            # In the previous code: U2[i1*2+j1, t]. Yes.
            
            u_vec = np.kron(U2[:, t], U2[:, s])
            # np.kron(a, b) where a is length 4 (i1*2+j1 order: 00, 01, 10, 11)
            # Result order: 0000, 0001, 0010, 0011, 0100... 
            # We need index i*4+j where i=2*i1+i2, j=2*j1+j2.
            # i*4+j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2.
            # np.kron(U2[:,t], U2[:,s]) index:
            # Let a_idx = i1*2+j1, b_idx = i2*2+j2.
            # Kron index = a_idx * 4 + b_idx = (i1*2+j1)*4 + (i2*2+j2) = 8*i1 + 4*j1 + 2*i2 + j2.
            # This is NOT 8*i1 + 4*i2 + 2*j1 + j2.
            # So simple kron doesn't work directly without permutation.
            
            # Let's just use the loop for safety and clarity, but structure it differently.
            for i1 in range(2):
                for i2 in range(2):
                    i = 2*i1 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = 2*j1 + j2
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
                            
            for j1 in range(2):
                for j2 in range(2):
                    j = 2*j1 + j2
                    for k1 in range(2):
                        for k2 in range(2):
                            k = 2*k1 + k2
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
                            
            for k1 in range(2):
                for k2 in range(2):
                    k = 2*k1 + k2
                    for i1 in range(2):
                        for i2 in range(2):
                            i = 2*i1 + i2
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
                            
    # Now build 4x4x8 (Rank 98)
    # It is two copies of 4x4x4.
    # U: (n*m, rank) = (16, 98)
    # V: (m*p, rank) = (32, 98)
    # W: (p*n, rank) = (32, 98)
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # First half: k in 0..3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i]
            
    # Second half: k in 4..7
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
