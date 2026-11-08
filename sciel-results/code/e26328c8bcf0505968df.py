DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # Try to construct a rank 48 solution
    # Based on R(4,4,2) <= 24
    # R(2,4,8) = R(4,2,8) = 2 * R(4,2,4)? No.
    # R(2,4,8) = R(4,8,2).
    # If R(4,4,2) = 24, then R(4,8,2) <= 2 * 24 = 48.
    
    # Laderman's 4x4x4 algorithm has rank 49.
    # We can restrict it to 4x4x2.
    
    # Hardcoded Laderman 4x4x4 factors
    # U, V, W for 4x4x4
    # We need to extract the part corresponding to 2 columns of the result.
    
    # This is complex to derive manually.
    # I will stick to the 56 solution but ensure it is robust.
    # Or try a simple 48 construction if I can find one.
    
    # For now, return the 56 solution.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, -1), (3, 1)],
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1)],
        [(2, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, -1)],
    ]
    
    def add_block(base_rank, i_off, j_off, k_off):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_off
                global_j = local_j + j_off
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_off
                global_k = local_k + k_off
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_off
                global_i = local_i + i_off
                W[global_k * 2 + global_i][current_rank] += val
    
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
