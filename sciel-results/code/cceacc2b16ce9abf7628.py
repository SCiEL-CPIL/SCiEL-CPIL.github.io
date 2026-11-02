import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 vectors
    # u[t] = [x00, x01, x10, x11]
    u_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 0, 1], # M2
        [1, 1, 0, 0], # M3
        [1, 0, 0, 0], # M4
        [0, 0, 1, 1], # M5
        [1, 0, -1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    # v[t] = [y00, y01, y10, y11]
    v_vecs = [
        [1, 0, 0, 1], # M1
        [1, 1, 0, 0], # M2
        [0, 0, 0, 1], # M3
        [0, 0, 1, -1], # M4
        [1, 0, 0, 0], # M5
        [0, 1, 0, 1], # M6
        [1, 0, 1, 0]  # M7
    ]
    # w[t] = [z00, z01, z10, z11]
    w_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [0, 1, 0, -1], # M3
        [1, 0, 0, 0], # M4
        [-1, 0, 1, 0], # M5
        [0, 1, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Top-level Strassen definitions
    # Each entry: (A_blocks, B_blocks, C_coeffs)
    # A_blocks: list of (block_id, coeff)
    # B_blocks: list of (block_id, coeff)
    # C_coeffs: [c11, c12, c21, c22]
    
    # Block IDs: 0=A11, 1=A12, 2=A21, 3=A22
    # For B: 0=B11, 1=B12, 2=B21, 3=B22
    
    top_level = [
        # M1 = (A11 + A22)(B11 + B22)
        { 'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [1, 0, 0, 1] },
        # M2 = A22(B11 + B12)
        { 'A': [(3, 1)], 'B': [(0, 1), (1, 1)], 'C': [0, 0, 1, 1] },
        # M3 = (A11 + A12)B22
        { 'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [0, 1, 0, -1] },
        # M4 = A11(B21 - B22)
        { 'A': [(0, 1)], 'B': [(2, 1), (3, -1)], 'C': [1, 0, 0, 0] },
        # M5 = (A21 + A22)B11
        { 'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [-1, 0, 1, 0] },
        # M6 = (A11 - A21)(B12 + B22)
        { 'A': [(0, 1), (2, -1)], 'B': [(1, 1), (3, 1)], 'C': [0, 1, 0, 1] },
        # M7 = (A12 - A22)(B21 + B11)
        { 'A': [(1, 1), (3, -1)], 'B': [(2, 1), (0, 1)], 'C': [1, 0, 0, 0] }
    ]
    
    # Mappings for A (4x4)
    # Local 2x2 indices 0,1,2,3 -> Global 16 indices
    a_maps = [
        [0, 1, 4, 5],   # A11
        [2, 3, 6, 7],   # A12
        [8, 9, 12, 13], # A21
        [10, 11, 14, 15] # A22
    ]
    
    # Mappings for B_L (4x4, cols 0-3 of 4x8)
    # Global B index = j * 8 + k
    bL_maps = [
        [0, 1, 8, 9],   # B11 (j=0,1, k=0,1) -> 0*8+0=0, 0*8+1=1, 1*8+0=8, 1*8+1=9
        [2, 3, 10, 11], # B12 (j=0,1, k=2,3) -> 0*8+2=2, 0*8+3=3, 1*8+2=10, 1*8+3=11
        [16, 17, 24, 25], # B21 (j=2,3, k=0,1) -> 2*8+0=16, 2*8+1=17, 3*8+0=24, 3*8+1=25
        [18, 19, 26, 27]  # B22 (j=2,3, k=2,3) -> 2*8+2=18, 2*8+3=19, 3*8+2=26, 3*8+3=27
    ]
    
    # Mappings for B_R (4x4, cols 4-7 of 4x8)
    bR_maps = [
        [4, 5, 12, 13],  # B11 (j=0,1, k=4,5) -> 0*8+4=4, 0*8+5=5, 1*8+4=12, 1*8+5=13
        [6, 7, 14, 15],  # B12 (j=0,1, k=6,7) -> 0*8+6=6, 0*8+7=7, 1*8+6=14, 1*8+7=15
        [20, 21, 28, 29], # B21 (j=2,3, k=4,5) -> 2*8+4=20, 2*8+5=21, 3*8+4=28, 3*8+5=29
        [22, 23, 30, 31]  # B22 (j=2,3, k=6,7) -> 2*8+6=22, 2*8+7=23, 3*8+6=30, 3*8+7=31
    ]
    
    # Mappings for C_L (4x4, cols 0-3 of 4x8)
    # Global W index = k * 4 + i
    cL_maps = [
        [0, 1, 4, 5],   # C11 (k=0,1, i=0,1) -> 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
        [2, 3, 6, 7],   # C12 (k=0,1, i=2,3) -> 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
        [8, 9, 12, 13], # C21 (k=2,3, i=0,1) -> 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
        [10, 11, 14, 15] # C22 (k=2,3, i=2,3) -> 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    ]
    
    # Mappings for C_R (4x4, cols 4-7 of 4x8)
    cR_maps = [
        [16, 17, 20, 21], # C11 (k=4,5, i=0,1) -> 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21
        [18, 19, 22, 23], # C12 (k=4,5, i=2,3) -> 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23
        [24, 25, 28, 29], # C21 (k=6,7, i=0,1) -> 6*4+0=24, 6*4+1=25, 7*4+0=28, 7*4+1=29
        [26, 27, 30, 31]  # C22 (k=6,7, i=2,3) -> 6*4+2=26, 6*4+3=27, 7*4+2=30, 7*4+3=31
    ]
    
    def embed(vec, maps, block_ids, coeffs, target_matrix, col):
        # vec: 4-element list
        # maps: list of 4 lists, each 4-element list of global indices
        # block_ids: list of block indices (0-3)
        # coeffs: list of coefficients corresponding to block_ids
        # target_matrix: 2D list
        # col: column index in target_matrix
        for bid, coeff in zip(block_ids, coeffs):
            if coeff == 0: continue
            m = maps[bid]
            for local_idx, val in enumerate(vec):
                if val != 0:
                    global_idx = m[local_idx]
                    target_matrix[global_idx][col] += val * coeff

    col = 0
    # Process C_L
    for tl in top_level:
        for t in range(7):
            # U
            embed(u_vecs[t], a_maps, [b[0] for b in tl['A']], [b[1] for b in tl['A']], U, col)
            # V
            embed(v_vecs[t], bL_maps, [b[0] for b in tl['B']], [b[1] for b in tl['B']], V, col)
            # W
            # tl['C'] is [c11, c12, c21, c22]
            embed(w_vecs[t], cL_maps, [0, 1, 2, 3], tl['C'], W, col)
            col += 1
            
    # Process C_R
    for tl in top_level:
        for t in range(7):
            # U
            embed(u_vecs[t], a_maps, [b[0] for b in tl['A']], [b[1] for b in tl['A']], U, col)
            # V
            embed(v_vecs[t], bR_maps, [b[0] for b in tl['B']], [b[1] for b in tl['B']], V, col)
            # W
            embed(w_vecs[t], cR_maps, [0, 1, 2, 3], tl['C'], W, col)
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
