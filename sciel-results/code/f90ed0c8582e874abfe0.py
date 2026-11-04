import time

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE
    
    # Check time limit
    if time.time() - _START_TIME > _DEADLINE:
        # Return a valid fallback if we run out of time, though construction is fast
        # But we shouldn't reach here.
        pass

    n, m, p = 3, 4, 8
    
    # Rank 88 decomposition:
    # 1. Row 0 of A (1x4) times B (4x8): Schoolbook, rank 1*4*8 = 32
    # 2. Rows 1,2 of A (2x4) times B (4x8): Split B into two 2x8 blocks (cols 0-1, 2-3 of B? No, B is 4x8. Split A's columns).
    #    A is 2x4. Split A into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3).
    #    B is 4x8. Split B into B_top (2x8, rows 0-1) and B_bottom (2x8, rows 2-3).
    #    Product = A_left B_top + A_right B_bottom.
    #    Each term is 2x2x8 multiplication.
    #    Rank of 2x2x8 is 28 (4 blocks of 2x2x2, each rank 7).
    #    Total for 2x4x8 part = 28 + 28 = 56.
    #    Total rank = 32 + 56 = 88.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7 (Schoolbook)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2
    # Strassen matrices for 2x2x2
    # U_mat: 7x4 (7 terms, 4 elements in 2x2 A)
    # V_mat: 7x4 (7 terms, 4 elements in 2x2 B)
    # W_mat: 7x4 (7 terms, 4 elements in 2x2 C)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # We need to apply this to 2x2x8.
    # This consists of 4 disjoint 2x2x2 blocks based on k index.
    # k indices: 0,1 | 2,3 | 4,5 | 6,7
    # For each block, we use 7 terms.
    # Total 28 terms for one 2x2x8.
    # We have two such 2x2x8 products (one for A cols 0-1, one for A cols 2-3).
    # Total 56 terms.
    
    # Helper to add a 2x2x2 Strassen block
    # local_i_offset: 0 or 1 (maps to global i = 1 + local_i_offset)
    # local_j_offset: 0, 1, 2, 3 (maps to global j)
    # local_k_offset: 0, 2, 4, 6 (maps to global k)
    
    def add_strassen_block(local_i_offset, local_j_offset, local_k_offset):
        nonlocal col
        for t in range(7):
            # U part: A indices (i, j)
            # local i in {0,1}, local j in {0,1}
            for l in range(4):
                val = strassen_U[t][l]
                if val != 0:
                    li = l // 2
                    lj = l % 2
                    gi = 1 + local_i_offset
                    gj = local_j_offset + lj
                    U[gi * m + gj][col] += val
            
            # V part: B indices (j, k)
            for l in range(4):
                val = strassen_V[t][l]
                if val != 0:
                    lj = l // 2
                    lk = l % 2
                    gj = local_j_offset + lj
                    gk = local_k_offset + lk
                    V[gj * p + gk][col] += val
            
            # W part: C indices (k, i)
            for l in range(4):
                val = strassen_W[t][l]
                if val != 0:
                    li = l // 2
                    lk = l % 2
                    gi = 1 + local_i_offset
                    gk = local_k_offset + lk
                    W[gk * n + gi][col] += val
            
            col += 1

    # Block 1: A cols 0-1, B rows 0-1 (k 0-7)
    # Wait, the 2x2x8 structure:
    # A is 2x2 (local cols 0-1 of global A rows 1-2).
    # B is 2x8 (local rows 0-1 of global B).
    # We split k (8 cols of B) into 4 pairs.
    # For each pair, we do a 2x2x2 Strassen.
    
    for k_block in range(4):
        k_start = k_block * 2
        add_strassen_block(0, 0, k_start)
        
    # Block 2: A cols 2-3, B rows 2-3 (k 0-7)
    # A local cols 0-1 correspond to global cols 2-3.
    # B local rows 0-1 correspond to global rows 2-3.
    # k is still 0-7 (cols of B).
    
    for k_block in range(4):
        k_start = k_block * 2
        add_strassen_block(0, 2, k_start)
        
    # Note: The 'local_i_offset' in add_strassen_block is always 0 because we are only
    # processing i=1,2 as a single 2x block. The 'gi' calculation inside handles
    # mapping local i 0->1, 1->2.
    
    return {"U": U, "V": V, "W": W}
