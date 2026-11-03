import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Rank 56 decomposition using 8 blocks of Strassen (rank 7)
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Strassen 2x2 * 2x2 matrices
    # U_local, V_local, W_local for one 2x2*2x2 multiplication
    # Indices: 0..3 correspond to a00, a01, a10, a11
    #          b00, b01, b10, b11
    #          c00, c01, c10, c11
    
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Mapping for the 8 blocks
    # A is 2x4. Split into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # B is 4x8. Split rows into B_top (2x8) and B_bot (2x8).
    # B_top splits into 4 blocks of 2x2: B_t0, B_t1, B_t2, B_t3.
    # B_bot splits into 4 blocks of 2x2: B_b0, B_b1, B_b2, B_b3.
    
    # Block (A1, B_tj) -> C_tj (2x2 part of C)
    # Block (A2, B_bj) -> C_bj (2x2 part of C)
    
    # U indices:
    # A1 rows 0-1, cols 0-1 -> U indices 0-3
    # A2 rows 0-1, cols 2-3 -> U indices 4-7
    
    # V indices:
    # B is 4x8.
    # B_top rows 0-1.
    #   B_t0 cols 0-1 -> V indices 0-3
    #   B_t1 cols 2-3 -> V indices 4-7
    #   B_t2 cols 4-5 -> V indices 8-11
    #   B_t3 cols 6-7 -> V indices 12-15
    # B_bot rows 2-3.
    #   B_b0 cols 0-1 -> V indices 16-19
    #   B_b1 cols 2-3 -> V indices 20-23
    #   B_b2 cols 4-5 -> V indices 24-27
    #   B_b3 cols 6-7 -> V indices 28-31
    
    # W indices:
    # C is 2x8.
    # C_top rows 0-1 (from A1*B_top? No, C = A1*B_top + A2*B_bot? 
    # Wait. A = [A1 A2]. B = [B_top; B_bot].
    # C = A1 B_top + A2 B_bot.
    # C is 2x8.
    # C_top is the top row of C? No, C has 2 rows.
    # Let's trace indices carefully.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # i in {0,1}. k in {0..7}. j in {0..3}.
    # A[i, 0..1] is A1. A[i, 2..3] is A2.
    # B[0..1, k] is B_top. B[2..3, k] is B_bot.
    # C[i, k] = sum_{j=0}^1 A1[i, j] B_top[j, k] + sum_{j=2}^3 A2[i, j-2] B_bot[j-2, k].
    # So C is the sum of two 2x8 matrices.
    # Let C1 = A1 B_top. C2 = A2 B_bot. C = C1 + C2.
    
    # We decompose C1 into 4 2x2 multiplications:
    # C1[:, 0:2] = A1 B_t0
    # C1[:, 2:4] = A1 B_t1
    # C1[:, 4:6] = A1 B_t2
    # C1[:, 6:8] = A1 B_t3
    
    # We decompose C2 into 4 2x2 multiplications:
    # C2[:, 0:2] = A2 B_b0
    # C2[:, 2:4] = A2 B_b1
    # C2[:, 4:6] = A2 B_b2
    # C2[:, 6:8] = A2 B_b3
    
    # Total 8 blocks.
    # Block 0: A1 B_t0 -> C1[:, 0:2]
    # Block 1: A1 B_t1 -> C1[:, 2:4]
    # Block 2: A1 B_t2 -> C1[:, 4:6]
    # Block 3: A1 B_t3 -> C1[:, 6:8]
    # Block 4: A2 B_b0 -> C2[:, 0:2]
    # Block 5: A2 B_b1 -> C2[:, 2:4]
    # Block 6: A2 B_b2 -> C2[:, 4:6]
    # Block 7: A2 B_b3 -> C2[:, 6:8]
    
    # U indices for A1: 0,1,2,3 (a00, a01, a10, a11)
    # U indices for A2: 4,5,6,7 (a00, a01, a10, a11 of A2)
    
    # V indices for B_t0: 0,1,2,3
    # V indices for B_t1: 4,5,6,7
    # V indices for B_t2: 8,9,10,11
    # V indices for B_t3: 12,13,14,15
    # V indices for B_b0: 16,17,18,19
    # V indices for B_b1: 20,21,22,23
    # V indices for B_b2: 24,25,26,27
    # V indices for B_b3: 28,29,30,31
    
    # W indices for C1[:, 0:2] (C[0:2, 0:2]):
    # W is p*n = 8*2 = 16.
    # W[k*n+i, r] convention? 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually W is (p*n, rank).
    # Index in W: k*n + i.
    # C is 2x8. i is row (0,1), k is col (0..7).
    # So W index for C[i, k] is k*2 + i.
    
    # C1[:, 0:2] corresponds to C[0:2, 0:2].
    # i=0, k=0 -> W[0]
    # i=0, k=1 -> W[2]
    # i=1, k=0 -> W[1]
    # i=1, k=1 -> W[3]
    
    # C1[:, 2:4] corresponds to C[0:2, 2:4].
    # i=0, k=2 -> W[4]
    # i=0, k=3 -> W[6]
    # i=1, k=2 -> W[5]
    # i=1, k=3 -> W[7]
    
    # C1[:, 4:6] -> W[8, 10, 9, 11]
    # C1[:, 6:8] -> W[12, 14, 13, 15]
    
    # C2 adds to the same C positions.
    # So W indices for C2 blocks are the SAME as C1 blocks.
    # Block 4 (A2 B_b0) adds to C[0:2, 0:2] -> W[0, 2, 1, 3]
    # Block 5 (A2 B_b1) adds to C[0:2, 2:4] -> W[4, 6, 5, 7]
    # etc.
    
    u_map_a1 = [0, 1, 2, 3]
    u_map_a2 = [4, 5, 6, 7]
    
    v_maps = [
        [0, 1, 2, 3],   # B_t0
        [4, 5, 6, 7],   # B_t1
        [8, 9, 10, 11], # B_t2
        [12, 13, 14, 15], # B_t3
        [16, 17, 18, 19], # B_b0
        [20, 21, 22, 23], # B_b1
        [24, 25, 26, 27], # B_b2
        [28, 29, 30, 31]  # B_b3
    ]
    
    w_maps = [
        [0, 2, 1, 3],   # C[:, 0:2]
        [4, 6, 5, 7],   # C[:, 2:4]
        [8, 10, 9, 11], # C[:, 4:6]
        [12, 14, 13, 15], # C[:, 6:8]
        [0, 2, 1, 3],   # C[:, 0:2] (added)
        [4, 6, 5, 7],   # C[:, 2:4] (added)
        [8, 10, 9, 11], # C[:, 4:6] (added)
        [12, 14, 13, 15] # C[:, 6:8] (added)
    ]
    
    # Fill U, V, W
    # 8 blocks, each 7 rank.
    for block_idx in range(8):
        base_rank = block_idx * 7
        u_map = u_map_a1 if block_idx < 4 else u_map_a2
        v_map = v_maps[block_idx]
        w_map = w_maps[block_idx]
        
        for l in range(7):
            r = base_rank + l
            # U
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[u_map[la], r] += val
            # V
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[v_map[lb], r] += val
            # W
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[w_map[lc], r] += val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
