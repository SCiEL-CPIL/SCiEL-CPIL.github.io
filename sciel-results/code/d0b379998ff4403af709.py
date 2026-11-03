import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 98
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # Returns list of 7 tuples: (u_coeffs, v_coeffs, w_coeffs)
    # u, v, w are lists of 4 floats
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),  # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # P3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # P5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # P7
    ]
    
    # Strassen 4x4 terms
    # Each term is (u_blocks, v_blocks, w_blocks)
    # u_blocks: dict mapping local 2x2 index (0..3) to list of (global_A_block, coeff)
    # Actually, easier to define mappings for the 4 local indices of the 2x2 inputs/outputs.
    
    # Global A indices for blocks:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # Global B indices for blocks (Left):
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # Global C indices for blocks (Left):
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    def get_block_indices(base, block_id):
        # block_id 0: 00, 1: 01, 2: 10, 3: 11
        # Returns list of 4 global indices
        if block_id == 0: return [base+0, base+1, base+4, base+5]
        if block_id == 1: return [base+2, base+3, base+6, base+7]
        if block_id == 2: return [base+8, base+9, base+12, base+13]
        if block_id == 3: return [base+10, base+11, base+14, base+15]

    # Define the 7 terms of 4x4 Strassen
    # u_map: local_idx -> list of (global_block_idx, coeff)
    # v_map: local_idx -> list of (global_block_idx, coeff)
    # w_map: local_idx -> list of (global_block_idx, coeff)
    
    terms_4x4 = [
        # M1 = (A00 + A11)(B00 + B11)
        {
            'u': {0: [(0, 1), (3, 1)], 1: [(0, 1), (3, 1)], 2: [(0, 1), (3, 1)], 3: [(0, 1), (3, 1)]},
            'v': {0: [(0, 1), (3, 1)], 1: [(0, 1), (3, 1)], 2: [(0, 1), (3, 1)], 3: [(0, 1), (3, 1)]},
            'w': {0: [(0, 1), (3, -1)], 1: [(0, 1), (3, -1)], 2: [(0, 1), (3, -1)], 3: [(0, 1), (3, -1)]}
        },
        # M2 = (A10 + A11)B00
        {
            'u': {0: [(2, 1), (3, 1)], 1: [(2, 1), (3, 1)], 2: [(2, 1), (3, 1)], 3: [(2, 1), (3, 1)]},
            'v': {0: [(0, 1)], 1: [(0, 1)], 2: [(0, 1)], 3: [(0, 1)]},
            'w': {0: [(2, 1), (3, -1)], 1: [(2, 1), (3, -1)], 2: [(2, 1), (3, -1)], 3: [(2, 1), (3, -1)]}
        },
        # M3 = A00(B01 - B11)
        {
            'u': {0: [(0, 1)], 1: [(0, 1)], 2: [(0, 1)], 3: [(0, 1)]},
            'v': {0: [(1, 1), (3, -1)], 1: [(1, 1), (3, -1)], 2: [(1, 1), (3, -1)], 3: [(1, 1), (3, -1)]},
            'w': {0: [(1, 1)], 1: [(1, 1)], 2: [(1, 1)], 3: [(1, 1)]}
        },
        # M4 = A11(B10 - B00)
        {
            'u': {0: [(3, 1)], 1: [(3, 1)], 2: [(3, 1)], 3: [(3, 1)]},
            'v': {0: [(2, 1), (0, -1)], 1: [(2, 1), (0, -1)], 2: [(2, 1), (0, -1)], 3: [(2, 1), (0, -1)]},
            'w': {0: [(0, 1), (2, 1)], 1: [(0, 1), (2, 1)], 2: [(0, 1), (2, 1)], 3: [(0, 1), (2, 1)]}
        },
        # M5 = (A00 + A01)B11
        {
            'u': {0: [(0, 1), (1, 1)], 1: [(0, 1), (1, 1)], 2: [(0, 1), (1, 1)], 3: [(0, 1), (1, 1)]},
            'v': {0: [(3, 1)], 1: [(3, 1)], 2: [(3, 1)], 3: [(3, 1)]},
            'w': {0: [(0, -1), (1, 1)], 1: [(0, -1), (1, 1)], 2: [(0, -1), (1, 1)], 3: [(0, -1), (1, 1)]}
        },
        # M6 = (A10 - A00)(B00 + B01)
        {
            'u': {0: [(2, 1), (0, -1)], 1: [(2, 1), (0, -1)], 2: [(2, 1), (0, -1)], 3: [(2, 1), (0, -1)]},
            'v': {0: [(0, 1), (1, 1)], 1: [(0, 1), (1, 1)], 2: [(0, 1), (1, 1)], 3: [(0, 1), (1, 1)]},
            'w': {0: [(3, 1)], 1: [(3, 1)], 2: [(3, 1)], 3: [(3, 1)]}
        },
        # M7 = (A01 - A11)(B10 + B11)
        {
            'u': {0: [(1, 1), (3, -1)], 1: [(1, 1), (3, -1)], 2: [(1, 1), (3, -1)], 3: [(1, 1), (3, -1)]},
            'v': {0: [(2, 1), (3, 1)], 1: [(2, 1), (3, 1)], 2: [(2, 1), (3, 1)], 3: [(2, 1), (3, 1)]},
            'w': {0: [(0, 1)], 1: [(0, 1)], 2: [(0, 1)], 3: [(0, 1)]}
        }
    ]
    
    col = 0
    # Process Left Half (B indices 0-15, C indices 0-15)
    for term in terms_4x4:
        for u_loc, v_loc, w_loc in strassen_2x2:
            u_glob = [0.0] * 16
            v_glob = [0.0] * 32
            w_glob = [0.0] * 32
            
            # Map U
            for local_idx in range(4):
                coeff_u = u_loc[local_idx]
                if coeff_u == 0: continue
                for block_id, coeff_b in term['u'][local_idx]:
                    idxs = get_block_indices(0, block_id) # A base 0
                    for k, idx in enumerate(idxs):
                        u_glob[idx] += coeff_u * coeff_b
                        
            # Map V
            for local_idx in range(4):
                coeff_v = v_loc[local_idx]
                if coeff_v == 0: continue
                for block_id, coeff_b in term['v'][local_idx]:
                    idxs = get_block_indices(0, block_id) # B base 0 (Left)
                    for k, idx in enumerate(idxs):
                        v_glob[idx] += coeff_v * coeff_b
                        
            # Map W
            for local_idx in range(4):
                coeff_w = w_loc[local_idx]
                if coeff_w == 0: continue
                for block_id, coeff_b in term['w'][local_idx]:
                    idxs = get_block_indices(0, block_id) # C base 0 (Left)
                    for k, idx in enumerate(idxs):
                        w_glob[idx] += coeff_w * coeff_b
                        
            for r in range(16): U[r][col] = u_glob[r]
            for r in range(32): V[r][col] = v_glob[r]
            for r in range(32): W[r][col] = w_glob[r]
            col += 1
            
    # Process Right Half (B indices 16-31, C indices 16-31)
    for term in terms_4x4:
        for u_loc, v_loc, w_loc in strassen_2x2:
            u_glob = [0.0] * 16
            v_glob = [0.0] * 32
            w_glob = [0.0] * 32
            
            # Map U (Same as before)
            for local_idx in range(4):
                coeff_u = u_loc[local_idx]
                if coeff_u == 0: continue
                for block_id, coeff_b in term['u'][local_idx]:
                    idxs = get_block_indices(0, block_id) # A base 0
                    for k, idx in enumerate(idxs):
                        u_glob[idx] += coeff_u * coeff_b
                        
            # Map V (Offset 16)
            for local_idx in range(4):
                coeff_v = v_loc[local_idx]
                if coeff_v == 0: continue
                for block_id, coeff_b in term['v'][local_idx]:
                    idxs = get_block_indices(16, block_id) # B base 16 (Right)
                    for k, idx in enumerate(idxs):
                        v_glob[idx] += coeff_v * coeff_b
                        
            # Map W (Offset 16)
            for local_idx in range(4):
                coeff_w = w_loc[local_idx]
                if coeff_w == 0: continue
                for block_id, coeff_b in term['w'][local_idx]:
                    idxs = get_block_indices(16, block_id) # C base 16 (Right)
                    for k, idx in enumerate(idxs):
                        w_glob[idx] += coeff_w * coeff_b
                        
            for r in range(16): U[r][col] = u_glob[r]
            for r in range(32): V[r][col] = v_glob[r]
            for r in range(32): W[r][col] = w_glob[r]
            col += 1
            
    return {"U": U, "V": V, "W": W}
