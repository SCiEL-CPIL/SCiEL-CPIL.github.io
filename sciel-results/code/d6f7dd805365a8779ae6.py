import time
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition patterns
    # Indices for 2x2: i,j,k in {0,1}
    # U, V, W are 4x7 (flattened)
    
    # r=0
    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    
    # r=1
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    
    # r=2
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    
    # r=3
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    
    # r=4
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    
    # r=5
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    
    # r=6
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]
    
    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    # 8 blocks of 2x2x2
    # A_L (i:0-1, j:0-1) * B_T (j:0-1, k:0-1)
    add_block(0, 0, 0, 0)
    # A_L (i:0-1, j:0-1) * B_T (j:0-1, k:2-3)
    add_block(7, 0, 0, 2)
    # A_L (i:0-1, j:0-1) * B_B (j:0-1, k:4-5)
    add_block(14, 0, 0, 4)
    # A_L (i:0-1, j:0-1) * B_B (j:0-1, k:6-7)
    add_block(21, 0, 0, 6)
    
    # A_R (i:0-1, j:2-3) * B_T (j:2-3, k:0-1)
    add_block(28, 0, 2, 0)
    # A_R (i:0-1, j:2-3) * B_T (j:2-3, k:2-3)
    add_block(35, 0, 2, 2)
    # A_R (i:0-1, j:2-3) * B_B (j:2-3, k:4-5)
    add_block(42, 0, 2, 4)
    # A_R (i:0-1, j:2-3) * B_B (j:2-3, k:6-7)
    add_block(49, 0, 2, 6)
    
    # Quick self-verification with random integer matrices
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A @ B directly
    C_direct = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    # Compute C via tensor decomposition: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_tensor = [[0]*p for _ in range(n)]
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(rank):
                    total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
            C_tensor[i][k] = total
    
    # Check correctness
    for i in range(n):
        for k in range(p):
            if C_direct[i][k] != C_tensor[i][k]:
                # Fallback: return trivial rank-64 decomposition
                rank_trivial = n * m * p
                U_t = [[0] * rank_trivial for _ in range(n * m)]
                V_t = [[0] * rank_trivial for _ in range(m * p)]
                W_t = [[0] * rank_trivial for _ in range(p * n)]
                idx = 0
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            U_t[i2*m + j2][idx] = 1
                            V_t[j2*p + k2][idx] = 1
                            W_t[k2*n + i2][idx] = 1
                            idx += 1
                return {"U": U_t, "V": V_t, "W": W_t}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
