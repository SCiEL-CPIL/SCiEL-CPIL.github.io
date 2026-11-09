import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Rank 23 decomposition for 3x3 matrix multiplication
    # Each term is a tuple: (u_map, v_map, w_map)
    # Keys are (row, col)
    laderman_terms = [
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (2,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1, (1,2): 1}, {(2,2): 1}, {(2,1): 1, (2,2): 1}),
        ({(2,2): 1}, {(2,0): 1, (0,0): -1}, {(0,2): 1, (0,0): -1}),
        ({(2,0): 1, (2,2): 1}, {(0,0): 1}, {(0,2): 1, (0,0): 1}),
        ({(0,2): 1, (2,2): 1}, {(2,0): 1}, {(0,0): 1, (0,2): 1}),
        ({(0,0): 1, (1,1): -1}, {(1,1): 1, (2,2): 1}, {(1,0): 1, (2,0): 1, (1,1): -1, (2,1): -1}),
        ({(1,1): 1, (2,2): -1}, {(2,2): 1, (0,0): 1}, {(2,1): 1, (0,1): 1, (2,2): -1, (0,2): -1}),
        ({(2,2): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}, {(0,2): 1, (1,2): 1, (0,0): -1, (1,0): -1}),
        ({(0,1): 1, (1,1): 1}, {(1,2): 1}, {(2,0): 1, (2,1): 1}),
        ({(1,1): 1, (2,1): 1}, {(2,1): 1}, {(1,1): 1, (1,2): 1}),
        ({(2,1): 1, (0,1): 1}, {(1,1): 1}, {(1,2): 1, (1,0): 1}),
        ({(0,2): 1, (1,2): 1}, {(2,0): 1}, {(0,0): 1, (0,1): 1}),
        ({(1,2): 1, (2,2): 1}, {(0,2): 1}, {(2,1): 1, (2,2): 1}),
        ({(2,2): 1, (0,2): 1}, {(2,0): 1}, {(0,0): 1, (0,2): 1}),
        ({(0,0): 1, (1,0): 1}, {(0,1): 1}, {(1,0): 1, (1,1): 1}),
        ({(1,0): 1, (2,0): 1}, {(1,2): 1}, {(2,0): 1, (2,1): 1}),
        ({(2,0): 1, (0,0): 1}, {(2,1): 1}, {(0,1): 1, (0,2): 1}),
        ({(0,1): 1, (1,0): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1, (2,0): 1}, {(2,1): 1}, {(2,0): 1, (2,1): 1}),
        ({(2,1): 1, (0,0): 1}, {(0,1): 1}, {(0,1): 1, (0,2): 1}),
        ({(0,2): 1, (1,0): 1}, {(2,0): 1}, {(0,0): 1, (0,1): 1}),
        ({(1,2): 1, (2,0): 1}, {(0,2): 1}, {(2,1): 1, (2,2): 1}),
    ]
    
    # Verify the Laderman terms by checking if they reconstruct the 3x3x3 tensor
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For 3x3, T[i,j,k] = 1 for all i,j,k in 0..2
    
    # Let's build the tensor from the terms and check
    T_check = np.zeros((3, 3, 3))
    for u_map, v_map, w_map in laderman_terms:
        for (i, j), u_val in u_map.items():
            for (j2, k), v_val in v_map.items():
                if j == j2: # j must match
                    for (k2, i2), w_val in w_map.items():
                        if k == k2 and i == i2: # k and i must match
                            T_check[i, j, k] += u_val * v_val * w_val
    
    # Check if T_check is all ones
    if not np.allclose(T_check, 1.0):
        # Fallback to schoolbook if Laderman terms are incorrect
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # If Laderman terms are correct, use them for the 3x3x6 problem
    # We split B into two 3x3 blocks: B1 (cols 0-2) and B2 (cols 3-5)
    # C = A * B = [A*B1, A*B2]
    # Rank(A*B1) + Rank(A*B2) = 23 + 23 = 46
    
    rank = 46
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For the first block (cols 0-2 of B, cols 0-2 of C)
    for r in range(23):
        u_map, v_map, w_map = laderman_terms[r]
        for (i, j), u_val in u_map.items():
            U[i * m + j, r] = u_val
        for (j, k), v_val in v_map.items():
            V[j * p + k, r] = v_val # k is 0,1,2 -> cols 0,1,2
        for (k, i), w_val in w_map.items():
            W[k * n + i, r] = w_val # k is 0,1,2 -> cols 0,1,2 of C
    
    # For the second block (cols 3-5 of B, cols 3-5 of C)
    for r in range(23):
        u_map, v_map, w_map = laderman_terms[r]
        col_offset = 23 + r
        for (i, j), u_val in u_map.items():
            U[i * m + j, col_offset] = u_val
        for (j, k), v_val in v_map.items():
            V[j * p + (k + 3), col_offset] = v_val # k is 0,1,2 -> cols 3,4,5
        for (k, i), w_val in w_map.items():
            W[(k + 3) * n + i, col_offset] = w_val # k is 0,1,2 -> cols 3,4,5 of C
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
