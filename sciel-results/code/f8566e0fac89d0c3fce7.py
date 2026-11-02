"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition using Strassen for 2x2x2 blocks
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    # U, V, W vectors of length 4 (local indices)
    # Local indices: 0:00, 1:01, 2:10, 3:11
    strassen_factors = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 0]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),
    ]
    
    # Schoolbook factors for 2x2x1
    # Local indices: A 00,01,10,11; B 00,10; C 0,1
    schoolbook_221_factors = [
        ([1, 0, 0, 0], [1, 0], [1, 0]), # A00*B00 -> C0
        ([0, 1, 0, 0], [0, 1], [1, 0]), # A01*B10 -> C0
        ([0, 0, 1, 0], [1, 0], [0, 1]), # A10*B00 -> C1
        ([0, 0, 0, 1], [0, 1], [0, 1]), # A11*B10 -> C1
    ]
    
    # Define sub-blocks
    # (j_offset, k_start, k_end, type)
    # j_offset: 0 for left block (j=0,1), 2 for right block (j=2,3)
    # k_start, k_end: range of k for this sub-block
    # type: 'strassen' or 'school'
    
    sub_blocks = [
        (0, 0, 2, 'strassen'), # L, k=0,1
        (0, 2, 4, 'strassen'), # L, k=2,3
        (0, 4, 6, 'strassen'), # L, k=4,5
        (0, 6, 7, 'school'),   # L, k=6
        (2, 0, 2, 'strassen'), # R, k=0,1
        (2, 2, 4, 'strassen'), # R, k=2,3
        (2, 4, 6, 'strassen'), # R, k=4,5
        (2, 6, 7, 'school'),   # R, k=6
    ]
    
    for j_off, k_start, k_end, stype in sub_blocks:
        if stype == 'strassen':
            for u_loc, v_loc, w_loc in strassen_factors:
                # Add to U
                for i in range(2):
                    for j in range(2):
                        local_j = i * 2 + j
                        val = u_loc[local_j]
                        if val != 0:
                            global_a_idx = i * m + (j_off + j)
                            U[global_a_idx][col] += val
                
                # Add to V
                for j in range(2):
                    for k in range(2):
                        local_k = j * 2 + k
                        val = v_loc[local_k]
                        if val != 0:
                            global_b_idx = (j_off + j) * p + (k_start + k)
                            V[global_b_idx][col] += val
                
                # Add to W
                for i in range(2):
                    for k in range(2):
                        local_w_idx = k * 2 + i
                        val = w_loc[local_w_idx]
                        if val != 0:
                            global_c_idx = (k_start + k) * n + i
                            W[global_c_idx][col] += val
                            
                col += 1
        else:
            for u_loc, v_loc, w_loc in schoolbook_221_factors:
                # Add to U
                for i in range(2):
                    for j in range(2):
                        local_j = i * 2 + j
                        val = u_loc[local_j]
                        if val != 0:
                            global_a_idx = i * m + (j_off + j)
                            U[global_a_idx][col] += val
                
                # Add to V
                for j in range(2):
                    # k is 0 for this sub-block (only 1 column)
                    val = v_loc[j]
                    if val != 0:
                        global_b_idx = (j_off + j) * p + k_start
                        V[global_b_idx][col] += val
                
                # Add to W
                for i in range(2):
                    val = w_loc[i]
                    if val != 0:
                        global_c_idx = k_start * n + i
                        W[global_c_idx][col] += val
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
