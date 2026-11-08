import time

DIMENSIONS = [2, 4, 8]
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if time.time() > _DEADLINE:
        return _build_strassen()
    # Rank is 56 based on 8 blocks of Strassen (rank 7)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition patterns
    # Indices for 2x2: i,j,k in {0,1}
    # Flattened indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # r=0
    U0 = [(0, 1), (3, 1)] # i=0,j=0; i=1,j=1
    V0 = [(0, 1), (3, 1)] # j=0,k=0; j=1,k=1
    W0 = [(0, 1), (3, 1)] # k=0,i=0; k=1,i=1
    
    # r=1
    U1 = [(2, 1), (3, 1)] # i=1,j=0; i=1,j=1
    V1 = [(0, 1)]         # j=0,k=0
    W1 = [(1, 1), (3, -1)] # k=0,i=1; k=1,i=1
    
    # r=2
    U2 = [(0, 1)]         # i=0,j=0
    V2 = [(1, 1), (3, -1)] # j=0,k=1; j=1,k=1
    W2 = [(2, 1), (3, 1)]  # k=1,i=0; k=1,i=1
    
    # r=3
    U3 = [(3, 1)]         # i=1,j=1
    V3 = [(2, 1), (0, -1)] # j=1,k=0; j=0,k=0
    W3 = [(0, 1), (1, 1)]  # k=0,i=0; k=0,i=1
    
    # r=4
    U4 = [(0, 1), (1, 1)] # i=0,j=0; i=0,j=1
    V4 = [(3, 1)]         # j=1,k=1
    W4 = [(0, -1), (2, 1)] # k=0,i=0; k=1,i=0
    
    # r=5
    U5 = [(2, 1), (0, -1)] # i=1,j=0; i=0,j=0
    V5 = [(0, 1), (1, 1)]  # j=0,k=0; j=0,k=1
    W5 = [(3, 1)]          # k=1,i=1
    
    # r=6
    U6 = [(1, 1), (3, -1)] # i=0,j=1; i=1,j=1
    V6 = [(2, 1), (3, 1)]  # j=1,k=0; j=1,k=1
    W6 = [(0, 1)]          # k=0,i=0
    
    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]
    
    # Helper to add a block
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * m + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * p + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * n + global_i][current_rank] += val

    # We need to cover i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    # We split j into two blocks of size 2: j in {0,1} and j in {2,3}
    # We split k into four blocks of size 2: k in {0,1}, {2,3}, {4,5}, {6,7}
    
    # Sub-problem 1: A_L (i:0-1, j:0-1) * B_T (j:0-1, k:0-7)
    # 4 blocks of 2x2x2
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    
    # Sub-problem 2: A_R (i:0-1, j:2-3) * B_B (j:2-3, k:0-7)
    # 4 blocks of 2x2x2
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}

def _build_strassen():
    """Fallback: same Strassen construction, extracted for deadline guard."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [(0, 1), (3, 1)], [(2, 1), (3, 1)], [(0, 1)], [(3, 1)],
        [(0, 1), (1, 1)], [(2, 1), (0, -1)], [(1, 1), (3, -1)]
    ]
    strassen_V = [
        [(0, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)], [(2, 1), (0, -1)],
        [(3, 1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)]
    ]
    strassen_W = [
        [(0, 1), (3, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)],
        [(0, -1), (2, 1)], [(3, 1)], [(0, 1)]
    ]
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                U[(local_i + i_offset) * m + (local_j + j_offset)][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                V[(local_j + j_offset) * p + (local_k + k_offset)][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                W[(local_k + k_offset) * n + (local_i + i_offset)][current_rank] += val
    
    add_block(0, 0, 0, 0); add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4); add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0); add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4); add_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
