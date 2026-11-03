def construct():
    n, m, p = 4, 4, 9
    
    # Total rank: 49 (4x4x4) + 49 (4x4x4) + 4 (4x4x1) = 102
    total_rank = 102
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 2x2x2 Strassen terms
    # U local: di*2+dj, V local: dj*2+dk, W local: dk*2+di
    s222 = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # M1
        ([(2,1),(3,1)], [(0,1)], [(1,1)]),              # M2
        ([(0,1)], [(1,1),(3,-1)], [(2,1)]),             # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),       # M4
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),       # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),       # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,-1)]),# M7
    ]
    
    # Level 1 Strassen for 4x4x4
    # A blocks: (br,bc) -> A[br*2:(br+1)*2, bc*2:(bc+1)*2]
    # B blocks: (br,bc) -> B[br*2:(br+1)*2, bc*2:(bc+1)*2]
    # Output blocks: (or,oc) -> C[or*2:(or+1)*2, oc*2:(oc+1)*2]
    
    # Level 1 terms: (A_block_coeffs, B_block_coeffs, output_block_coeffs)
    # where output_block_coeffs maps (or,oc) -> coeff
    l1 = [
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        # M2: (A21+A22)B11 -> C21 += 1
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3: A11(B12-B22) -> C12 += 1
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}),
        # M4: A22(B21-B11) -> C11 += 1, C21 += 1
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        # M6: (A21-A11)(B11+B12) -> C22 += 1
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        # M7: (A12-A22)(B21+B22) -> C11 += 1, C22 -= 1
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1,(1,1):-1}),
    ]
    
    def add_4x4x4_block(p_offset, rank_offset):
        """Add 49 rank terms for a 4x4x4 block with B columns starting at p_offset."""
        for l1_idx in range(7):
            a_c, b_c, w_c = l1[l1_idx]
            for l2_idx in range(7):
                r = rank_offset + l1_idx * 7 + l2_idx
                u_l2, v_l2, w_l2 = s222[l2_idx]
                
                # U: level-2 (di,dj) -> level-1 blocks
                for (lu, cu) in u_l2:
                    di, dj = divmod(lu, 2)
                    for (br, bc), ca in a_c.items():
                        U[(br*2+di)*m + (bc*2+dj)][r] += ca * cu
                
                # V: level-2 (dj,dk) -> level-1 blocks, with p_offset
                for (lv, cv) in v_l2:
                    dj, dk = divmod(lv, 2)
                    for (br, bc), cb in b_c.items():
                        V[(br*2+dj)*p + (p_offset + bc*2+dk)][r] += cb * cv
                
                # W: level-2 (dk,di) -> output blocks
                for (lw, cw) in w_l2:
                    dk, di = divmod(lw, 2)
                    for (or_, oc), co in w_c.items():
                        W[(p_offset + oc*2+dk)*n + (or_*2+di)][r] += co * cw
    
    # 4x4x4 block 1: columns 0-3
    add_4x4x4_block(0, 0)
    # 4x4x4 block 2: columns 4-7
    add_4x4x4_block(4, 49)
    
    # 4x4x1: column 8, rank 4
    for j in range(4):
        r = 98 + j
        for i in range(4):
            U[i*m + j][r] = 1
            V[j*p + 8][r] = 1
            W[8*n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
