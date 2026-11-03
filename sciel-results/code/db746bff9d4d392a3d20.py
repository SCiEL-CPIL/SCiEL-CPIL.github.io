import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # Target Rank: 88
    # 32 for M(1,4,8)
    # 56 for M(2,4,8) -> 4 * M(2,2,4) -> 4 * 14
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: M(1,4,8) ---
    # n=1, m=4, p=8
    # i=2 (global row index for this block)
    i_base = 2
    for j in range(4):
        for k in range(8):
            U[i_base * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i_base][col] = 1
            col += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    # --- Part 2: M(2,4,8) ---
    # Decomposed into 4 sub-problems of M(2,2,4)
    # Each M(2,2,4) uses M(2,2,2) (Strassen) x M(1,1,2)
    
    # Strassen Coefficients (u, v, w) for M(2,2,2)
    # Flattened 2x2 vectors
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, -C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # M(1,1,2) Coefficients
    # x is size 1, y is size 2, z is size 2
    inner_x = [1, 1]
    inner_y = [[1, 0], [0, 1]]
    inner_z = [[1, 0], [0, 1]]
    
    # Sub-problem definitions: (i_base, j_base, k_base)
    # i_base: row offset in A (and C)
    # j_base: col offset in A (and row offset in B)
    # k_base: col offset in B (and C)
    
    sub_problems = [
        (0, 0, 0), # A[0:2, 0:2] * B[0:2, 0:4] -> C[0:2, 0:4]
        (0, 0, 4), # A[0:2, 0:2] * B[0:2, 4:8] -> C[0:2, 4:8]
        (0, 2, 0), # A[0:2, 2:4] * B[2:4, 0:4] -> C[0:2, 0:4]
        (0, 2, 4)  # A[0:2, 2:4] * B[2:4, 4:8] -> C[0:2, 4:8]
    ]
    
    for i_base, j_base, k_base in sub_problems:
        for s in range(7):
            u_s = strassen_u[s]
            v_s = strassen_v[s]
            w_s = strassen_w[s]
            
            for t in range(2):
                x_t = inner_x[t]
                y_t = inner_y[t]
                z_t = inner_z[t]
                
                # Construct local vectors via Kronecker product
                # U_loc: size 4 (2x2)
                # V_loc: size 4 (2x2)
                # W_loc: size 4 (2x2)
                
                # U_loc[i_loc * 2 + j_loc] = u_s[i_loc * 2 + j_loc] * x_t
                # Note: u_s is flattened 2x2.
                # x_t is scalar.
                
                # Map to Global U
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u_s[i_loc * 2 + j_loc] * x_t
                        if val != 0:
                            i_glob = i_base + i_loc
                            j_glob = j_base + j_loc
                            U[i_glob * m + j_glob][col] += val
                
                # Map to Global V
                # v_s is for B (2x2). y_t is for B (1x2).
                # V_loc[j_loc * 2 + k_loc] = v_s[j_loc * 2 + k_strassen] * y_t[k_loc]?
                # Wait, B in M(2,2,4) is 2x4.
                # Indices j_loc (0..1), k_loc (0..1) from inner dim.
                # The "strassen" k index in v_s corresponds to the 2nd dim of the 2x2 block.
                # The "inner" k index in y_t corresponds to the split of the 4 cols.
                # So global k = k_base + k_strassen * 2 + k_inner?
                # No.
                # M(2,2,2) B is 2x2. Cols 0,1.
                # M(1,1,2) B is 1x2. Cols 0,1.
                # Product B is 2x4.
                # The columns are interleaved or blocked?
                # Standard Kronecker product of matrices:
                # If B1 is 2x2 and B2 is 1x2.
                # B1 (kron) B2 is 2x4.
                # Row i of B1, Col j of B1 becomes a 1x2 block from B2.
                # So Col (j * 2 + k_inner).
                # Thus k_glob = k_base + j_strassen * 2 + k_inner.
                
                for j_loc in range(2):
                    for k_strassen in range(2):
                        val_v_s = v_s[j_loc * 2 + k_strassen]
                        for k_inner in range(2):
                            val_y_t = y_t[k_inner]
                            val = val_v_s * val_y_t
                            if val != 0:
                                j_glob = j_base + j_loc
                                k_glob = k_base + k_strassen * 2 + k_inner
                                V[j_glob * p + k_glob][col] += val
                
                # Map to Global W
                # w_s is for C (2x2). z_t is for C (1x2).
                # C in M(2,2,4) is 2x4.
                # i_glob = i_base + i_loc.
                # k_glob = k_base + k_strassen * 2 + k_inner.
                
                for i_loc in range(2):
                    for k_strassen in range(2):
                        val_w_s = w_s[i_loc * 2 + k_strassen]
                        for k_inner in range(2):
                            val_z_t = z_t[k_inner]
                            val = val_w_s * val_z_t
                            if val != 0:
                                i_glob = i_base + i_loc
                                k_glob = k_base + k_strassen * 2 + k_inner
                                W[k_glob * n + i_glob][col] += val
                                
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                    
    return {"U": U, "V": V, "W": W}

