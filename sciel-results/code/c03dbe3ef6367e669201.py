import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Block size 2, number of blocks 3
    block_size = 2
    num_blocks = 3
    
    # Schoolbook decomposition for 3x3x3
    rank_3 = num_blocks ** 3
    U3 = np.zeros((num_blocks * num_blocks, rank_3))
    V3 = np.zeros((num_blocks * num_blocks, rank_3))
    W3 = np.zeros((num_blocks * num_blocks, rank_3))
    
    col = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                U3[I * num_blocks + J, col] = 1
                V3[J * num_blocks + K, col] = 1
                W3[K * num_blocks + I, col] = 1
                col += 1
                
    # Strassen decomposition for 2x2x2
    rank_2 = 7
    U2 = np.zeros((block_size * block_size, rank_2))
    V2 = np.zeros((block_size * block_size, rank_2))
    W2 = np.zeros((block_size * block_size, rank_2))
    
    # Strassen formulas
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11 contributions? No, W is for C.
    # W index is c*n + a.
    # C00 gets M1. C11 gets M1.
    # C00 is (c=0, a=0) -> W[0, 0]
    # C11 is (c=1, a=1) -> W[2+1, 0] = W[3, 0]
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10 gets M2, C11 gets -M2
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01 gets M3, C11 gets M3
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00 gets M4, C10 gets M4
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00 gets -M5, C01 gets M5
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11 gets M6
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00 gets M7

    total_rank = rank_3 * rank_2
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for t in range(rank_3):
        for s in range(rank_2):
            r = t * rank_2 + s
            # Combine U
            for I in range(num_blocks):
                for J in range(num_blocks):
                    u3 = U3[I * num_blocks + J, t]
                    if u3 != 0:
                        for a in range(block_size):
                            for b in range(block_size):
                                u2 = U2[a * block_size + b, s]
                                if u2 != 0:
                                    i = I * block_size + a
                                    j = J * block_size + b
                                    U[i * m + j, r] += u3 * u2
                                    
            # Combine V
            for J in range(num_blocks):
                for K in range(num_blocks):
                    v3 = V3[J * num_blocks + K, t]
                    if v3 != 0:
                        for b in range(block_size):
                            for c in range(block_size):
                                v2 = V2[b * block_size + c, s]
                                if v2 != 0:
                                    j = J * block_size + b
                                    k = K * block_size + c
                                    V[j * p + k, r] += v3 * v2
                                    
            # Combine W
            for K in range(num_blocks):
                for I in range(num_blocks):
                    w3 = W3[K * num_blocks + I, t]
                    if w3 != 0:
                        for c in range(block_size):
                            for a in range(block_size):
                                w2 = W2[c * block_size + a, s]
                                if w2 != 0:
                                    k = K * block_size + c
                                    i = I * block_size + a
                                    W[k * n + i, r] += w3 * w2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
