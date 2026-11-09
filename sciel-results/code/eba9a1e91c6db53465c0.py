import numpy as np

def construct():
    n, m, p = 5, 5, 5
    rank = 110
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    # 2x2 Strassen terms (flattened 4-vectors)
    # Order: (0,0), (0,1), (1,0), (1,1)
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 1, 0]), # M4
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, -1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # 4x4 Strassen terms
    # Each term is a tuple: (A_blocks, B_blocks, C_blocks)
    # Blocks are 2x2.
    # A_blocks: list of (block_id, coeff)
    # Block IDs: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
    
    def get_block_indices(block_id, n_sub=2):
        # Returns list of global indices for U (A), V (B), W (C)
        # A and B are 4x4 in the top-left of 5x5.
        # So indices are 0..3 for rows/cols.
        # U index: i*5 + j
        # V index: j*5 + k
        # W index: k*5 + i
        
        r0, r1, c0, c1 = 0, 0, 0, 0
        if block_id == 0: r0, r1, c0, c1 = 0, 1, 0, 1
        elif block_id == 1: r0, r1, c0, c1 = 0, 1, 2, 3
        elif block_id == 2: r0, r1, c0, c1 = 2, 3, 0, 1
        elif block_id == 3: r0, r1, c0, c1 = 2, 3, 2, 3
        
        u_indices = []
        v_indices = []
        w_indices = []
        for i in range(r0, r1 + 1):
            for j in range(c0, c1 + 1):
                u_indices.append(i * 5 + j)
                v_indices.append(j * 5 + i) # Wait, B is 4x4. B_{jk}. j is row, k is col.
                # For B, the block is B_{r0..r1, c0..c1}.
                # So v_index is j*5 + k where j is row, k is col.
                # In the block, row is i, col is j?
                # Let's stick to (row, col) for the block.
                # Block (0,0) of B is rows 0,1 cols 0,1.
                # So v_indices for block (0,0) are (0,0), (0,1), (1,0), (1,1).
                # Indices: 0*5+0, 0*5+1, 1*5+0, 1*5+1.
                # My loop above: i is row, j is col.
                # So v_index should be i*5 + j.
                # But wait, V is indexed by (j, k) for B_{jk}.
                # In the 4x4 block, B is 4x4.
                # So V index is row*5 + col.
                # So yes, i*5 + j.
                
                # For W, C is 4x4.
                # C_{ki}? No, W index is k*5 + i.
                # C is 4x4. C_{row, col}.
                # So W index is col*5 + row?
                # Task says W[k*n+i].
                # So if C is 5x5, W index for C_{ik} is k*5 + i.
                # In the 4x4 block, C is 4x4.
                # So W index for C_{ik} is k*5 + i.
                # Here i is row, k is col.
                # So W index is col*5 + row.
                # In my loop, i is row, j is col.
                # So W index is j*5 + i.
                
                w_indices.append(j * 5 + i)
        
        return u_indices, v_indices, w_indices

    # Precompute block indices
    block_indices = [get_block_indices(b) for b in range(4)]
    
    # 4x4 Strassen definitions
    # (A_blocks, B_blocks, C_blocks)
    strassen_4x4 = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),         # M2 (C10, C11)
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),         # M3 (C01, C11)
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),         # M4 (C00, C10)
        ([(0, 1), (2, 1)], [(3, 1)], [(0, -1), (1, 1)]),         # M5 (C00, C01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),         # M6 (C11)
        ([(3, 1), (1, -1)], [(1, 1), (3, 1)], [(0, 1)])          # M7 (C00)
    ]
    
    # Generate 49 terms
    for a_blocks, b_blocks, c_blocks in strassen_4x4:
        for u2, v2, w2 in strassen_2x2:
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for block_id, coeff_a in a_blocks:
                u_idx, v_idx, w_idx = block_indices[block_id]
                for k in range(4):
                    u_val = u2[k] * coeff_a
                    if u_val != 0:
                        u_dict[u_idx[k]] = u_dict.get(u_idx[k], 0) + u_val
            
            for block_id, coeff_b in b_blocks:
                u_idx, v_idx, w_idx = block_indices[block_id]
                for k in range(4):
                    v_val = v2[k] * coeff_b
                    if v_val != 0:
                        v_dict[v_idx[k]] = v_dict.get(v_idx[k], 0) + v_val
            
            for block_id, coeff_c in c_blocks:
                u_idx, v_idx, w_idx = block_indices[block_id]
                for k in range(4):
                    w_val = w2[k] * coeff_c
                    if w_val != 0:
                        w_dict[w_idx[k]] = w_dict.get(w_idx[k], 0) + w_val
            
            add_term(u_dict, v_dict, w_dict)
            
    # Boundary terms
    # A01 B10 (4x4)
    for i in range(4):
        for k in range(4):
            add_term({i * 5 + 4: 1}, {4 * 5 + k: 1}, {k * 5 + i: 1})
            
    # A00 B01 (16)
    for i in range(4):
        for j in range(4):
            add_term({i * 5 + j: 1}, {j * 5 + 4: 1}, {4 * 5 + i: 1})
            
    # A10 B00 (16)
    for j in range(4):
        for k in range(4):
            add_term({4 * 5 + j: 1}, {j * 5 + k: 1}, {k * 5 + 4: 1})
            
    # A01 B11 (4)
    for i in range(4):
        add_term({i * 5 + 4: 1}, {4 * 5 + 4: 1}, {4 * 5 + i: 1})
        
    # A11 B10 (4)
    for k in range(4):
        add_term({4 * 5 + 4: 1}, {4 * 5 + k: 1}, {k * 5 + 4: 1})
        
    # A10 B01 (4)
    for j in range(4):
        add_term({4 * 5 + j: 1}, {j * 5 + 4: 1}, {4 * 5 + 4: 1})
        
    # A11 B11 (1)
    add_term({4 * 5 + 4: 1}, {4 * 5 + 4: 1}, {4 * 5 + 4: 1})
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
