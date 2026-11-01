import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition for 2x3 * 3x8
    # Part 1: M_{2,1,8} rank 16
    # Part 2: M_{2,2,8} rank 28 (4 blocks of Strassen 2x2x2)
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: j=0 term
    # C[i, k] += A[i, 0] * B[0, k]
    # i in 0..1, k in 0..7
    r = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, r] = 1
            V[0 * p + k, r] = 1
            W[k * n + i, r] = 1
            r += 1
            
    # Part 2: j=1,2 terms
    # A' = A[:, 1:3] (2x2), B' = B[1:3, :] (2x8)
    # Split B' into 4 blocks of 2x2: B'_t = B[1:3, 2t:2t+2]
    # Strassen on A' * B'_t
    # A' indices in U: 
    # a11 -> A[0,1] -> idx 1
    # a12 -> A[0,2] -> idx 2
    # a21 -> A[1,1] -> idx 4
    # a22 -> A[1,2] -> idx 5
    # B'_t indices in V:
    # b11_t -> B[1, 2t] -> idx 8 + 2t
    # b12_t -> B[1, 2t+1] -> idx 9 + 2t
    # b21_t -> B[2, 2t] -> idx 16 + 2t
    # b22_t -> B[2, 2t+1] -> idx 17 + 2t
    # C'_t indices in W:
    # c11_t -> C[0, 2t] -> idx 4t
    # c12_t -> C[0, 2t+1] -> idx 4t + 2
    # c21_t -> C[1, 2t] -> idx 4t + 1
    # c22_t -> C[1, 2t+1] -> idx 4t + 3
    
    # Strassen coefficients
    # s=1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -> a11+a22, b11+b22, c11+c22
    # s=2: u=(0,0,1,1), v=(1,0,0,0), w=(0,1,-1,0) -> a21+a22, b11, c21-c22 (wait, c21+=1, c22-=1)
    # s=3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,1) -> a11, b12-b22, c12+c22
    # s=4: u=(0,0,0,1), v=(0,0,1,-1), w=(1,1,0,0) -> a22, b21-b11, c11+c21
    # s=5: u=(1,1,0,0), v=(0,0,0,1), w=(-1,0,1,0) -> a11+a12, b22, -c11+c12
    # s=6: u=(-1,0,1,0), v=(1,1,0,0), w=(0,0,0,1) -> -a11+a21, b11+b12, c22
    # s=7: u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,0) -> a12-a22, b21+b22, c11
    
    # u_vectors: indices [a11, a12, a21, a22] -> U indices [1, 2, 4, 5]
    u_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    u_indices = [1, 2, 4, 5]
    
    # v_vectors: indices [b11, b12, b21, b22]
    v_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # w_vectors: indices [c11, c12, c21, c22]
    # Note: W index order for a block is [c11, c21, c12, c22] in my previous mapping?
    # Let's check:
    # c11 -> 4t
    # c21 -> 4t+1
    # c12 -> 4t+2
    # c22 -> 4t+3
    # So the vector indices [0,1,2,3] correspond to [c11, c21, c12, c22].
    w_vecs = [
        [1, 0, 0, 1], # c11 + c22
        [0, 1, 0, -1], # c21 - c22
        [0, 0, 1, 1], # c12 + c22
        [1, 1, 0, 0], # c11 + c21
        [-1, 0, 1, 0], # -c11 + c12
        [0, 0, 0, 1], # c22
        [1, 0, 0, 0]  # c11
    ]
    
    for t in range(4):
        for s in range(7):
            r = 16 + t * 7 + s
            
            # Fill U
            for idx, val in zip(u_indices, u_vecs[s]):
                if val != 0:
                    U[idx, r] = val
                    
            # Fill V
            # b11 -> 8+2t, b12 -> 9+2t, b21 -> 16+2t, b22 -> 17+2t
            v_indices = [8 + 2*t, 9 + 2*t, 16 + 2*t, 17 + 2*t]
            for idx, val in zip(v_indices, v_vecs[s]):
                if val != 0:
                    V[idx, r] = val
                    
            # Fill W
            # c11 -> 4t, c21 -> 4t+1, c12 -> 4t+2, c22 -> 4t+3
            w_indices = [4*t, 4*t+1, 4*t+2, 4*t+3]
            for idx, val in zip(w_indices, w_vecs[s]):
                if val != 0:
                    W[idx, r] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
