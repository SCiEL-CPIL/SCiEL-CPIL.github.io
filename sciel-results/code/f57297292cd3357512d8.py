import time
import sys

DIMENSIONS = [4, 4, 4]

def construct():
    # Rank 49 decomposition using Recursive Strassen
    # Time guard
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # 1. Define the rank-7 decomposition for 2x2 matrix multiplication
    # u, v, w are lists of 7 vectors, each of length 4
    # Indices: 0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2)
    u2 = [
        [1, 0, 0, 1],  # P1: (a11+a22)
        [0, 0, 1, 1],  # P2: (a21+a22)
        [1, 0, 0, 0],  # P3: a11
        [0, 0, 0, 1],  # P4: a22
        [1, 1, 0, 0],  # P5: (a11+a12)
        [-1, 0, 1, 0], # P6: (a21-a11)
        [0, 1, 0, -1]  # P7: (a12-a22)
    ]
    v2 = [
        [1, 0, 0, 1],  # P1: (b11+b22)
        [1, 0, 0, 0],  # P2: b11
        [0, 1, 0, -1], # P3: (b12-b22)
        [-1, 0, 1, 0], # P4: (b21-b11)
        [0, 0, 0, 1],  # P5: b22
        [1, 1, 0, 0],  # P6: (b11+b12)
        [0, 0, 1, 1]   # P7: (b21+b22)
    ]
    w2 = [
        [1, 0, 0, 1],  # P1: c11 + c22
        [0, 0, 1, -1], # P2: c21 - c22
        [0, 1, 0, 1],  # P3: c12 + c22
        [1, 0, 1, 0],  # P4: c11 + c21
        [-1, 1, 0, 0], # P5: -c11 + c12
        [0, 0, 0, 1],  # P6: c22
        [1, 0, 0, 0]   # P7: c11
    ]
    
    # 2. Define the Strassen block combinations for 4x4 matrices
    # Blocks: 0:11, 1:12, 2:21, 3:22
    # X_k = sum alpha_k[block] * A_block
    # Y_k = sum beta_k[block] * B_block
    # C_block = sum gamma_k[block] * M_k
    
    # Step k: 0..6
    # alpha_k: coefficients for A blocks
    # beta_k: coefficients for B blocks
    # gamma_k: coefficients for C blocks (how M_k contributes to C blocks)
    
    alpha = [
        [1, 0, 0, 1],  # k=0 (P1): A11 + A22
        [0, 0, 1, 1],  # k=1 (P2): A21 + A22
        [1, 0, 0, 0],  # k=2 (P3): A11
        [0, 0, 0, 1],  # k=3 (P4): A22
        [1, 1, 0, 0],  # k=4 (P5): A11 + A12
        [-1, 0, 1, 0], # k=5 (P6): -A11 + A21
        [0, 1, 0, -1]  # k=6 (P7): A12 - A22
    ]
    
    beta = [
        [1, 0, 0, 1],  # k=0 (P1): B11 + B22
        [1, 0, 0, 0],  # k=1 (P2): B11
        [0, 1, 0, -1], # k=2 (P3): B12 - B22
        [-1, 0, 1, 0], # k=3 (P4): -B11 + B21
        [0, 0, 0, 1],  # k=4 (P5): B22
        [1, 1, 0, 0],  # k=5 (P6): B11 + B12
        [0, 0, 1, 1]   # k=6 (P7): B21 + B22
    ]
    
    # Gamma: how M_k contributes to C blocks (C11, C12, C21, C22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    gamma = [
        [1, 0, 0, 1],  # k=0 (M1): C11 + C22
        [0, 0, 1, -1], # k=1 (M2): C21 - C22
        [0, 1, 0, 1],  # k=2 (M3): C12 + C22
        [1, 0, 1, 0],  # k=3 (M4): C11 + C21
        [-1, 1, 0, 0], # k=4 (M5): -C11 + C12
        [0, 0, 0, 1],  # k=5 (M6): C22
        [1, 0, 0, 0]   # k=6 (M7): C11
    ]
    
    # 3. Construct global U, V, W
    # U shape: (16, 49)
    # V shape: (16, 49)
    # W shape: (16, 49)
    
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Mapping of global indices to block and local index
    # Global index 0..15
    # Block 0 (11): 0,1,2,3 -> local 0,1,2,3
    # Block 1 (12): 4,5,6,7 -> local 0,1,2,3
    # Block 2 (21): 8,9,10,11 -> local 0,1,2,3
    # Block 3 (22): 12,13,14,15 -> local 0,1,2,3
    
    def get_block_local(idx):
        return idx // 4, idx % 4
    
    col = 0
    for k in range(7):
        for r in range(7):
            # For each element in A (16 elements)
            for i in range(16):
                block_a, local_a = get_block_local(i)
                # Coefficient for A[i] in the r-th component of the k-th Strassen step
                # u2[r][local_a] is the coeff for the local element in the 2x2 matrix X_k
                # alpha[k][block_a] is the coeff for the block in X_k
                U[i][col] = u2[r][local_a] * alpha[k][block_a]
                
            # For each element in B (16 elements)
            for j in range(16):
                block_b, local_b = get_block_local(j)
                V[j][col] = v2[r][local_b] * beta[k][block_b]
                
            # For each element in C (16 elements)
            # Note: W convention is W[k*n+i, r] in task description?
            # "output factor follows the official trace convention W[k*n+i,r]"
            # Usually W is indexed by output elements.
            # Let's assume W index is the output element index.
            for c_idx in range(16):
                block_c, local_c = get_block_local(c_idx)
                # w2[r][local_c] is the coeff for the local element in the 2x2 matrix M_k
                # gamma[k][block_c] is the coeff for the block in the final C
                W[c_idx][col] = w2[r][local_c] * gamma[k][block_c]
                
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
