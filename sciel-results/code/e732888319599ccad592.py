"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build rank-7 decomposition for 2x2x2 tensor (Strassen)
    # T[i,j,k] for i,j,k in {0,1}
    # U2: (2*2, 7) = (4, 7), V2: (2*2, 7) = (4, 7), W2: (2*2, 7) = (4, 7)
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for 2x2x2
    U2 = np.array([
        [1,  0,  0,  0,  0,  0,  0],  # i=0,j=0
        [0,  1,  0,  0,  0,  0,  0],  # i=0,j=1
        [0,  0,  1,  1,  0,  0,  0],  # i=1,j=0
        [0,  0,  1,  1,  1,  0,  0],  # i=1,j=1
    ], dtype=float)
    V2 = np.array([
        [1,  0,  0,  0,  0,  0,  0],  # j=0,k=0
        [0,  1,  0,  0,  0,  0,  0],  # j=0,k=1
        [0,  0,  1,  1,  1,  1,  0],  # j=1,k=0
        [0,  0,  1,  1,  1,  1,  1],  # j=1,k=1
    ], dtype=float)
    W2 = np.array([
        [1,  0,  0,  0,  0,  0,  0],  # k=0,i=0
        [0,  0,  0,  0,  1,  1,  1],  # k=0,i=1
        [0,  1,  0,  0,  1,  1,  1],  # k=1,i=0
        [0,  0,  1,  1,  1,  1,  1],  # k=1,i=1
    ], dtype=float)
    
    # Build rank-4 decomposition for 2x2x1 tensor
    # T[i,j,k] for i,j in {0,1}, k in {0}
    # U221: (2*2, 4) = (4, 4), V221: (2*1, 4) = (2, 4), W221: (1*2, 4) = (2, 4)
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for 2x2x1 (n=2,m=2,p=1)
    U221 = np.array([
        [1, 0, 0, 0],  # i=0,j=0
        [0, 1, 0, 0],  # i=0,j=1
        [0, 0, 1, 0],  # i=1,j=0
        [0, 0, 0, 1],  # i=1,j=1
    ], dtype=float)
    V221 = np.array([
        [1, 0, 0, 0],  # j=0,k=0
        [0, 1, 0, 0],  # j=1,k=0
    ], dtype=float)
    W221 = np.array([
        [1, 0, 0, 0],  # k=0,i=0
        [0, 0, 1, 0],  # k=0,i=1
    ], dtype=float)
    
    # Build rank-2 decomposition for 2x1x1 tensor
    # T[i,j,k] for i in {0,1}, j in {0}, k in {0}
    # U211: (1*1, 2) = (1, 2), V211: (1*1, 2) = (1, 2), W211: (1*2, 2) = (2, 2)
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for 2x1x1 (n=2,m=1,p=1)
    U211 = np.array([[1, 0]], dtype=float)  # i=0,j=0 and i=1,j=0 -> only one row? No...
    # Wait, for 2x1x1: n=2, m=1, p=1
    # U shape: (n*m, rank) = (2*1, 2) = (2, 2)
    # V shape: (m*p, rank) = (1*1, 2) = (1, 2)
    # W shape: (p*n, rank) = (1*2, 2) = (2, 2)
    # T[i,j,k] = E_{i,k} for the 2x1x1 tensor
    U211 = np.array([
        [1, 0],  # i=0,j=0
        [0, 1],  # i=1,j=0
    ], dtype=float)
    V211 = np.array([
        [1, 1],  # j=0,k=0
    ], dtype=float)
    W211 = np.array([
        [1, 0],  # k=0,i=0
        [0, 1],  # k=0,i=1
    ], dtype=float)
    
    # Now assemble the full 4x4x5 decomposition
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2 (k in 0..3) or k=4
    
    # Part 1: k in {0,1,2,3}, decomposed as k = 2*k1 + k2
    # T[i,j,k] = T2[i1,j1,k1] * T221[i2,j2,k2] for k2 in {0,1}
    # Rank = 7 * 4 = 28
    
    # Part 2: k = 4, which is like a 4x4x1 strip
    # T[i,j,4] = T441[i,j,0]
    # 4x4x1 can be decomposed as (2x2x1) ⊗ (2x1x1)
    # Rank = 4 * 2 = 8
    
    # Total rank = 28 + 8 = 36
    
    rank1 = 7 * 4  # 28
    rank2 = 4 * 2  # 8
    total_rank = rank1 + rank2  # 36
    
    U = np.zeros((n * m, total_rank), dtype=float)  # (16, 36)
    V = np.zeros((m * p, total_rank), dtype=float)  # (20, 36)
    W = np.zeros((p * n, total_rank), dtype=float)  # (20, 36)
    
    # Part 1: k in {0,1,2,3}
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # U index: i*m + j = (2*i1+i2)*4 + (2*j1+j2)
    # V index: j*p + k = (2*j1+j2)*5 + (2*k1+k2)
    # W index: k*n + i = (2*k1+k2)*4 + (2*i1+i2)
    
    for c1 in range(7):
        for c2 in range(4):
            c = c1 * 4 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    i = 2 * i1 + i2
                                    j = 2 * j1 + j2
                                    k = 2 * k1 + k2
                                    # U contribution
                                    u_idx = i * m + j
                                    v_idx = j * p + k
                                    w_idx = k * n + i
                                    U[u_idx, c] += U2[i1 * 2 + j1, c1] * U221[i2 * 2 + j2, c2]
                                    V[v_idx, c] += V2[j1 * 2 + k1, c1] * V221[j2 * 1 + k2, c2]
                                    W[w_idx, c] += W2[k1 * 2 + i1, c1] * W221[k2 * 2 + i2, c2]
    
    # Part 2: k = 4
    # 4x4x1: i = 2*i1 + i2, j = 2*j1 + j2
    # T[i,j,4] = T221[i2, j2, 0] * T211[i1, 0, 0]
    # Wait, let me reconsider. For 4x4x1:
    # U shape: (4*4, 8) = (16, 8), V shape: (4*1, 8) = (4, 8), W shape: (1*4, 8) = (4, 8)
    # Decompose as (2x2x1) ⊗ (2x1x1):
    # i = 2*i1 + i2, j = 2*j1 + j2
    # U441[i,j,c] = U221[i2,j2,c2] * U211[i1,0,c1]
    # V441[j,4,c] = V221[j2,0,c2] * V211[0,0,c1]  (since k=4 is the only k)
    # W441[4,i,c] = W221[0,i2,c2] * W211[0,i1,c1]
    
    for c1 in range(4):
        for c2 in range(2):
            c = rank1 + c1 * 2 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            k = 4
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            U[u_idx, c] += U221[i2 * 2 + j2, c1] * U211[i1, c2]
                            V[v_idx, c] += V221[j2, c1] * V211[0, c2]
                            W[w_idx, c] += W221[i2, c1] * W211[i1, c2]
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
