"""Program for matrix multiplication tensor decomposition (4,5,5)."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s wall-clock limit
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Structured Kronecker/Block approach
    # Partition indices into blocks
    # i: [0,1], [2,3]
    # j: [0,1], [2,3], [4]
    # k: [0,1], [2,3], [4]
    
    i_blocks = [(0, 2), (2, 4)]
    j_blocks = [(0, 2), (2, 4), (4, 5)]
    k_blocks = [(0, 2), (2, 4), (4, 5)]
    
    # Calculate total rank
    total_rank = 0
    block_ranks = []
    for i_start, i_end in i_blocks:
        ni = i_end - i_start
        for j_start, j_end in j_blocks:
            nj = j_end - j_start
            for k_start, k_end in k_blocks:
                nk = k_end - k_start
                if ni == 2 and nj == 2 and nk == 2:
                    r = 7
                else:
                    r = ni * nj * nk
                total_rank += r
                block_ranks.append((i_start, i_end, j_start, j_end, k_start, k_end, r))
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_s, i_e, j_s, j_e, k_s, k_e, r in block_ranks:
        if r == 7:
            # Strassen decomposition for 2x2x2
            # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
            # Global indices: i = i_s + i_loc, etc.
            
            # U entries (ni x nj) -> (2 x 2)
            # V entries (nj x nk) -> (2 x 2)
            # W entries (nk x ni) -> (2 x 2)
            
            # Strassen matrices U_s, V_s, W_s
            # Indices in local block:
            # U: [i_loc * 2 + j_loc]
            # V: [j_loc * 2 + k_loc]
            # W: [k_loc * 2 + i_loc]
            
            U_s = [
                [1, 0, 0, 1],  # M1
                [0, 0, 1, 1],  # M2
                [1, 0, 0, 0],  # M3
                [0, 0, 0, 1],  # M4
                [1, 0, 1, 0],  # M5
                [-1, 1, 0, 0], # M6
                [0, 0, -1, 1]  # M7
            ]
            V_s = [
                [1, 0, 0, 1],  # M1
                [1, 0, 0, 0],  # M2
                [0, 1, 0, -1], # M3
                [0, 0, 1, -1], # M4
                [0, 0, 0, 1],  # M5
                [1, 0, 1, 0],  # M6
                [0, 1, 0, 1]   # M7
            ]
            W_s = [
                [1, 0, 0, 1],  # M1: C00+1, C11+1
                [0, 0, 1, -1], # M2: C10+1, C11-1
                [0, 1, 0, 1],  # M3: C01+1, C11+1
                [1, 0, 1, 0],  # M4: C00+1, C10+1
                [-1, 1, 0, 0], # M5: C00-1, C01+1
                [0, 0, 0, 1],  # M6: C11+1
                [1, 0, 0, 0]   # M7: C00+1
            ]
            
            for s in range(7):
                # Fill U
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = U_s[s][i_loc * 2 + j_loc]
                        if val != 0:
                            gi = i_s + i_loc
                            gj = j_s + j_loc
                            U[gi * m + gj][col] += val
                            
                # Fill V
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = V_s[s][j_loc * 2 + k_loc]
                        if val != 0:
                            gj = j_s + j_loc
                            gk = k_s + k_loc
                            V[gj * p + gk][col] += val
                            
                # Fill W
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = W_s[s][k_loc * 2 + i_loc]
                        if val != 0:
                            gk = k_s + k_loc
                            gi = i_s + i_loc
                            W[gk * n + gi][col] += val
                            
                col += 1
        else:
            # Schoolbook for this block
            for i_loc in range(i_e - i_s):
                for j_loc in range(j_e - j_s):
                    for k_loc in range(k_e - k_s):
                        gi = i_s + i_loc
                        gj = j_s + j_loc
                        gk = k_s + k_loc
                        
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1

    # Verify the result
    def verify(U, V, W, tol=1e-6):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        rank = U_np.shape[1]
        
        T_decomp = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            U_r = U_np[:, r].reshape(n, m)
            V_r = V_np[:, r].reshape(m, p)
            W_r = W_np[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
        
        T_target = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(T_decomp, T_target, atol=tol, rtol=tol)
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        return best_result
# EVOLVE-BLOCK-END
