"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    block_size = 2
    
    # Strassen-like terms for 2x3x2
    # Rank 11
    strassen_terms = [
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(0,1), (1,1)], [1, -1]),
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1]),
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1]),
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (1,0)], [-1, 1]),
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
        ([(0,2)], [1], [(2,0)], [1], [(0,0)], [1]),
        ([(0,2)], [1], [(2,1)], [1], [(1,0)], [1]),
        ([(1,2)], [1], [(2,0)], [1], [(0,1)], [1]),
        ([(1,2)], [1], [(2,1)], [1], [(1,1)], [1]),
    ]
    
    rank_per_block = len(strassen_terms)
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for b in range(num_blocks):
        for term in strassen_terms:
            u_idx, u_val, v_idx, v_val, w_idx, w_val = term
            for (i, j), val in zip(u_idx, u_val):
                U[i * m + j][col_offset] += val
            for (j, k_local), val in zip(v_idx, v_val):
                k_global = b * block_size + k_local
                V[j * p + k_global][col_offset] += val
            for (k_local, i), val in zip(w_idx, w_val):
                k_global = b * block_size + k_local
                W[k_global * n + i][col_offset] += val
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
