import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    # Helper to create zero vectors
    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)

    # 1. Strassen 4x4x4 for first 4 columns (cols 0-3)
    # This generates 49 terms.
    # We define the 49 terms explicitly based on the recursive Strassen structure.
    # A 4x4x4 Strassen is composed of 7 outer Strassen terms, each of which
    # is a 2x2x4 * 2x2x4? No, it's 2x2 blocks.
    # Outer: 7 terms. Each term involves 2x2 blocks.
    # Inner: Each 2x2 block multiplication is 7 terms.
    # Total 7*7 = 49.
    
    # Indices for 2x2 blocks in 4x4 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Outer Strassen terms for 4x4 * 4x4:
    # Each term is defined by coefficients for A-blocks, B-blocks, C-blocks.
    # A-blocks: 0,1,2,3. B-blocks: 0,1,2,3. C-blocks: 0,1,2,3.
    # The 7 Strassen terms for 2x2 * 2x2 are:
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    # M2 = (A10+A11)B00 -> C10 - C11
    # M3 = A00(B01-B11) -> C01 + C11
    # M4 = A11(B10-B00) -> C00 + C10
    # M5 = (A00+A01)B11 -> -C00 + C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    # We need to map these to the 4x4 level.
    # The "2x2" blocks at the outer level are actually 2x2 sub-matrices of the 4x4.
    # Let's call the 4x4 matrix A. It has 4 blocks of size 2x2: A00, A01, A10, A11.
    # The Strassen formulas apply to these blocks.
    # So for M1:
    # U part: A00 + A11. This means in the U vector, we add 1 to all indices in A00 and A11.
    # V part: B00 + B11.
    # W part: C00 + C11.
    # But wait, the multiplication is of the blocks.
    # The product of block A00 (2x2) and block B00 (2x2) is a 2x2 matrix.
    # This product is placed in C00.
    # So we need to expand the block multiplication into scalar multiplications.
    # The block multiplication A00 * B00 is itself a 2x2 * 2x2 multiplication.
    # So we use the 7 Strassen terms for 2x2 * 2x2 for each pair of blocks involved.
    
    # This is what the current code does with `strassen_4x4_head`.
    # I will replicate this logic but cleaner.
    
    outer_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # Each entry is (block_idx, coeff)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]), # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]), # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]), # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]), # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]), # M7
    ]
    
    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]), # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]), # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]), # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]), # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]), # M7
    ]
    
    def get_idx_A(block, local):
        br, bc = block // 2, block % 2
        lr, lc = local // 2, local % 2
        r, c = br*2+lr, bc*2+lc
        return r*m + c
        
    def get_idx_B(block, local):
        br, bc = block // 2, block % 2
        lr, lc = local // 2, local % 2
        r, c = br*2+lr, bc*2+lc
        return r*p + c
        
    def get_idx_C(block, local):
        br, bc = block // 2, block % 2
        lr, lc = local // 2, local % 2
        r, c = br*2+lr, bc*2+lc
        return c*n + r # W[k*n+i] -> col*n + row

    for oa, ob, oc in outer_terms:
        for ia, ib, ic in inner_terms:
            u = zero_u()
            v = zero_v()
            w = zero_w()
            
            for (ba, ca), (la, da) in zip(oa, ia):
                # This zip is incorrect in the original logic.
                # Original logic: for x_local, x_coeff in inner_x: for a_block, a_coeff in outer_a
                # It iterates all combinations.
                pass
            
            # Correct logic from original:
            for (la, da) in ia:
                for (ba, ca) in oa:
                    idx = get_idx_A(ba, la)
                    u[idx] += ca * da
            
            for (lb, db) in ib:
                for (bb, cb) in ob:
                    idx = get_idx_B(bb, lb)
                    v[idx] += cb * db
            
            for (lc, dc) in ic:
                for (bc, cc) in oc:
                    idx = get_idx_C(bc, lc)
                    w[idx] += cc * dc
            
            add_term(u, v, w)
            
    # 2. Strassen 4x4x2 for cols 4-5
    # 4 blocks of 2x2 * 2x2
    # Each block: A(i,j) 2x2, B(i, 4-5) 2x2, C(i, 4-5) 2x2
    # Use 7 Strassen terms for each 2x2 * 2x2.
    def strassen_2x2_block(i_off, j_off, k_off):
        # i_off, j_off, k_off are top-left corners
        # A block: rows i_off..i_off+1, cols j_off..j_off+1
        # B block: rows j_off..j_off+1, cols k_off..k_off+1
        # C block: rows i_off..i_off+1, cols k_off..k_off+1
        
        # 7 terms
        # M1
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + j_off] = 1; u[(i_off+1)*m + (j_off+1)] = 1
        v[j_off*p + k_off] = 1; v[(j_off+1)*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1; w[(k_off+1)*n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M2
        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off+1)*m + j_off] = 1; u[(i_off+1)*m + (j_off+1)] = 1
        v[j_off*p + k_off] = 1
        w[k_off*n + (i_off+1)] = 1; w[(k_off+1)*n + (i_off+1)] = -1
        add_term(u, v, w)
        
        # M3
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + j_off] = 1
        v[j_off*p + (k_off+1)] = 1; v[(j_off+1)*p + (k_off+1)] = -1
        w[(k_off+1)*n + i_off] = 1; w[(k_off+1)*n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M4
        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off+1)*m + (j_off+1)] = 1
        v[(j_off+1)*p + k_off] = 1; v[j_off*p + k_off] = -1
        w[k_off*n + i_off] = 1; w[k_off*n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M5
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + j_off] = 1; u[i_off*m + (j_off+1)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+1)*n + i_off] = 1; w[k_off*n + i_off] = -1
        add_term(u, v, w)
        
        # M6
        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off+1)*m + j_off] = 1; u[i_off*m + j_off] = -1
        v[j_off*p + k_off] = 1; v[j_off*p + (k_off+1)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M7
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off*m + (j_off+1)] = 1; u[(i_off+1)*m + (j_off+1)] = -1
        v[(j_off+1)*p + k_off] = 1; v[(j_off+1)*p + (k_off+1)] = 1
        w[k_off*n + i_off] = 1
        add_term(u, v, w)

    for i_b in range(2):
        for j_b in range(2):
            strassen_2x2_block(i_b*2, j_b*2, 4)
            
    # 3. Schoolbook 4x4x1 for col 6
    for i in range(n):
        for j in range(m):
            u = zero_u(); v = zero_v(); w = zero_w()
            u[i*m + j] = 1
            v[j*p + 6] = 1
            w[6*n + i] = 1
            add_term(u, v, w)
            
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
