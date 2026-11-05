import time
import sys

DIMENSIONS = [4, 4, 5]
_START_TIME = time.time()
TIME_LIMIT = 120.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    if _time_remaining() < 1.0:
        # Fallback: Schoolbook rank 80
        rank = 80
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Strassen 4x4x4 decomposition (Rank 49)
    # Split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Super-Strassen terms M1..M7
    # Each Mi is a 2x2x2 tensor (Rank 7)
    
    # Mapping of blocks to global indices
    # A blocks: 0:A00, 1:A01, 2:A10, 3:A11
    # B blocks: 0:B00, 1:B01, 2:B10, 3:B11
    # C blocks: 0:C00, 1:C01, 2:C10, 3:C11
    
    # Coefficients for A in each Super Term
    # M1: A00+A11 -> [1, 0, 0, 1]
    # M2: A10+A11 -> [0, 0, 1, 1]
    # M3: A00     -> [1, 0, 0, 0]
    # M4: A11     -> [0, 0, 0, 1]
    # M5: A00+A01 -> [1, 1, 0, 0]
    # M6: A10-A00 -> [-1, 0, 1, 0]
    # M7: A01-A11 -> [0, 1, 0, -1]
    a_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Coefficients for B in each Super Term
    # M1: B00+B11 -> [1, 0, 0, 1]
    # M2: B00     -> [1, 0, 0, 0]
    # M3: B01-B11 -> [0, 1, 0, -1]
    # M4: B10-B00 -> [-1, 0, 1, 0]
    # M5: B11     -> [0, 0, 0, 1]
    # M6: B00+B01 -> [1, 1, 0, 0]
    # M7: B10+B11 -> [0, 0, 1, 1]
    b_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Contribution of each Super Term to C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # c_contrib[super_idx][c_block] = coeff
    # c_block: 0:C00, 1:C01, 2:C10, 3:C11
    c_contrib = [
        [1, 0, 0, 1],  # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, C11
        [0, 1, 0, 1],  # M3 -> C01, C11
        [1, 0, 1, 0],  # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> C00, C01
        [0, 0, 0, 1],  # M6 -> C11
        [1, 0, 0, 0]   # M7 -> C00
    ]
    
    # Inner Strassen 2x2x2 (Rank 7)
    # P = [[P00, P01], [P10, P11]]
    # Q = [[Q00, Q01], [Q10, Q11]]
    # R = P Q
    
    # Inner terms s0..s6
    # s0 = (P00+P11)(Q00+Q11)
    # s1 = (P10+P11)Q00
    # s2 = P00(Q01-Q11)
    # s3 = P11(Q10-Q00)
    # s4 = (P00+P01)Q11
    # s5 = (P10-P00)(Q00+Q01)
    # s6 = (P01-P11)(Q10+Q11)
    
    # P coeffs for inner terms (indices 0:00, 1:01, 2:10, 3:11)
    # s0: P00, P11 -> [1, 0, 0, 1]
    # s1: P10, P11 -> [0, 0, 1, 1]
    # s2: P00      -> [1, 0, 0, 0]
    # s3: P11      -> [0, 0, 0, 1]
    # s4: P00, P01 -> [1, 1, 0, 0]
    # s5: P10, -P00-> [-1, 0, 1, 0]
    # s6: P01, -P11-> [0, 1, 0, -1]
    p_inner_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Q coeffs for inner terms
    # s0: Q00, Q11 -> [1, 0, 0, 1]
    # s1: Q00      -> [1, 0, 0, 0]
    # s2: Q01, -Q11-> [0, 1, 0, -1]
    # s3: Q10, -Q00-> [-1, 0, 1, 0]
    # s4: Q11      -> [0, 0, 0, 1]
    # s5: Q00, Q01 -> [1, 1, 0, 0]
    # s6: Q10, Q11 -> [0, 0, 1, 1]
    q_inner_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # Contribution of inner terms to local R blocks (0:00, 1:01, 2:10, 3:11)
    # R00 = s0 + s3 - s4 + s6
    # R01 = s2 + s4
    # R10 = s1 + s3
    # R11 = s0 - s1 + s2 + s5
    r_inner_contrib = [
        [1, 0, 0, 1],  # s0 -> R00, R11
        [0, 0, 1, -1], # s1 -> R10, R11
        [0, 1, 0, 1],  # s2 -> R01, R11
        [1, 0, 1, 0],  # s3 -> R00, R10
        [-1, 1, 0, 0], # s4 -> R00, R01
        [0, 0, 0, 1],  # s5 -> R11
        [1, 0, 0, 0]   # s6 -> R00
    ]
    
    # Helper to get global indices for a block
    def get_block_indices(block_type, block_idx, local_r, local_c):
        # block_type: 'A', 'B', 'C'
        # block_idx: 0..3
        # local_r, local_c: 0..1
        if block_type == 'A':
            # A00: 0,0. A01: 0,1. A10: 1,0. A11: 1,1
            br = block_idx // 2
            bc = block_idx % 2
            gi = br * 2 + local_r
            gj = bc * 2 + local_c
            return (gi, gj)
        elif block_type == 'B':
            # B00: 0,0. B01: 0,1. B10: 1,0. B11: 1,1
            br = block_idx // 2
            bc = block_idx % 2
            gj = br * 2 + local_r
            gk = bc * 2 + local_c
            return (gj, gk)
        elif block_type == 'C':
            # C00: 0,0. C01: 0,1. C10: 1,0. C11: 1,1
            br = block_idx // 2
            bc = block_idx % 2
            gi = br * 2 + local_r
            gk = bc * 2 + local_c
            return (gi, gk)

    for super_idx in range(7):
        for inner_idx in range(7):
            u_list = []
            v_list = []
            w_list = []
            
            # Construct U (A factors)
            # P is formed by summing A blocks with coeffs from a_coeffs[super_idx]
            # Inner term inner_idx uses P elements with coeffs from p_inner_coeffs[inner_idx]
            # P_elem_idx corresponds to local (r,c) of the 2x2 P matrix
            # P_elem 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            
            # For each P element involved in inner_idx:
            for p_elem_idx, p_coeff in enumerate(p_inner_coeffs[inner_idx]):
                if p_coeff == 0: continue
                # Get local r,c for this P element
                pr = p_elem_idx // 2
                pc = p_elem_idx % 2
                # P is sum of A blocks.
                # For each A block involved in super_idx:
                for a_block_idx, a_coeff in enumerate(a_coeffs[super_idx]):
                    if a_coeff == 0: continue
                    # The A block contributes to P at local (pr, pc)
                    # Global A indices:
                    gi, gj = get_block_indices('A', a_block_idx, pr, pc)
                    u_list.append((gi, gj, a_coeff * p_coeff))
            
            # Construct V (B factors)
            for q_elem_idx, q_coeff in enumerate(q_inner_coeffs[inner_idx]):
                if q_coeff == 0: continue
                qr = q_elem_idx // 2
                qc = q_elem_idx % 2
                for b_block_idx, b_coeff in enumerate(b_coeffs[super_idx]):
                    if b_coeff == 0: continue
                    gj, gk = get_block_indices('B', b_block_idx, qr, qc)
                    v_list.append((gj, gk, b_coeff * q_coeff))
            
            # Construct W (C factors)
            # The local result R (2x2) is added to global C blocks determined by super_idx.
            # c_contrib[super_idx] gives coeffs for C blocks 0..3.
            # r_inner_contrib[inner_idx] gives coeffs for local R blocks 0..3.
            # Local R block 0 is C00, 1 is C01, 2 is C10, 3 is C11.
            # Wait, the mapping is:
            # Super term super_idx adds its local 2x2 result to specific global 2x2 blocks.
            # c_contrib[super_idx][c_block_idx] is the coefficient.
            # For each such global block, the local result R is added.
            # R itself is composed of inner terms.
            # r_inner_contrib[inner_idx][r_block_idx] is the coefficient of this inner term in local R block r_block_idx.
            # So for each global block c_block_idx:
            #   sign_super = c_contrib[super_idx][c_block_idx]
            #   For each local block r_block_idx:
            #      sign_inner = r_inner_contrib[inner_idx][r_block_idx]
            #      total_coeff = sign_super * sign_inner
            #      The global block c_block_idx corresponds to C block c_block_idx.
            #      The local block r_block_idx corresponds to position (r_block_idx//2, r_block_idx%2) within that C block?
            #      NO. The local result R is a 2x2 matrix.
            #      R[0,0] goes to C_block[0,0]. R[0,1] goes to C_block[0,1].
            #      So local block idx 0 (R00) maps to local (0,0) of the global block.
            #      local block idx 1 (R01) maps to local (0,1) of the global block.
            #      local block idx 2 (R10) maps to local (1,0) of the global block.
            #      local block idx 3 (R11) maps to local (1,1) of the global block.
            
            for c_block_idx, super_coeff in enumerate(c_contrib[super_idx]):
                if super_coeff == 0: continue
                # Determine global block offset
                # c_block_idx 0: C00 (rows 0-1, cols 0-1)
                # c_block_idx 1: C01 (rows 0-1, cols 2-3)
                # c_block_idx 2: C10 (rows 2-3, cols 0-1)
                # c_block_idx 3: C11 (rows 2-3, cols 2-3)
                
                for r_block_idx, inner_coeff in enumerate(r_inner_contrib[inner_idx]):
                    if inner_coeff == 0: continue
                    total_coeff = super_coeff * inner_coeff
                    if total_coeff == 0: continue
                    
                    # Local position within the 2x2 C block
                    lr = r_block_idx // 2
                    lc = r_block_idx % 2
                    
                    # Global position
                    gi, gk = get_block_indices('C', c_block_idx, lr, lc)
                    # W index is k*n + i
                    w_list.append((gk, gi, total_coeff))
            
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 (Rank 16) for column 4
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
