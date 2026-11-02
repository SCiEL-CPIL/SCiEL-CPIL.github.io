import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 49 for 4x4 matrix multiplication
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define 2x2 Strassen factors (U2, V2, W2)
    # Rows: local indices 0..3 (row-major: 00, 01, 10, 11)
    # Cols: local rank 0..6
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 1, 0, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Active blocks for each Strassen product s (0-indexed)
    # Blocks are (row_block, col_block)
    active_A = [
        [(0,0), (1,1)], # M1
        [(1,0), (1,1)], # M2
        [(0,0)],        # M3
        [(1,1)],        # M4
        [(0,0), (0,1)], # M5
        [(1,0), (0,0)], # M6
        [(0,1), (1,1)]  # M7
    ]
    
    active_B = [
        [(0,0), (1,1)], # M1
        [(0,0)],        # M2
        [(0,1), (1,1)], # M3
        [(1,0), (0,0)], # M4
        [(1,1)],        # M5
        [(0,0), (0,1)], # M6
        [(1,0), (1,1)]  # M7
    ]
    
    # C blocks are determined by the output structure, 
    # but since W2 already encodes the linear combination for each local C index,
    # we just need to map the global C index to the local C index and block.
    # However, W2 is applied to the local 2x2 block.
    # The global C is composed of 4 blocks.
    # For a given global C index z, it belongs to a specific block (br, bc).
    # The value W[z, R] should be W2[l, t] if the block (br, bc) is "active" for the output?
    # Actually, every block of C receives contributions from all M_s?
    # No, the Strassen algorithm computes the blocks of C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # This means for block C00, only M1, M4, M5, M7 contribute.
    # For block C01, only M3, M5 contribute.
    # etc.
    
    # So we need active_C for each block.
    active_C = {
        (0,0): [0, 3, 4, 6], # M1, M4, M5, M7 (indices 0, 3, 4, 6)
        (0,1): [2, 4],       # M3, M5
        (1,0): [1, 3],       # M2, M4
        (1,1): [0, 1, 2, 5]  # M1, M2, M3, M6
    }
    
    # Helper to get local index from global index
    def get_local_index(global_idx, dim):
        # dim is 4
        r = global_idx // dim
        c = global_idx % dim
        br = r // 2
        bc = c // 2
        lr = r % 2
        lc = c % 2
        local_idx = 2 * lr + lc
        return br, bc, local_idx

    # Fill U
    for x in range(16):
        br, bc, l = get_local_index(x, 4)
        for s in range(7):
            if (br, bc) in active_A[s]:
                for t in range(7):
                    R = 7 * s + t
                    U[x][R] = U2[l][t]
                    
    # Fill V
    for y in range(16):
        br, bc, l = get_local_index(y, 4)
        for s in range(7):
            if (br, bc) in active_B[s]:
                for t in range(7):
                    R = 7 * s + t
                    V[y][R] = V2[l][t]
                    
    # Fill W
    for z in range(16):
        # z = k * n + i. n=4.
        # i = z % 4, k = z // 4
        i = z % 4
        k = z // 4
        br = i // 2
        bc = k // 2
        lr = i % 2
        lc = k % 2
        l = 2 * lr + lc
        
        # Get the list of active s for this C block
        active_s_for_block = active_C[(br, bc)]
        for s in active_s_for_block:
            for t in range(7):
                R = 7 * s + t
                W[z][R] = W2[l][t]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
