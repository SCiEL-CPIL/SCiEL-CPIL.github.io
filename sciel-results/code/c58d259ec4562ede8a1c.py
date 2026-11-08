import time

def construct():
    n, m, p = 4, 4, 5
    # We will build the 4x4x4 part with rank 49 and 4x4x1 part with rank 16.
    # Total rank 65.
    
    U = [[0] * 65 for _ in range(16)]
    V = [[0] * 65 for _ in range(20)]
    W = [[0] * 65 for _ in range(20)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Strassen 2x2x2 terms (7 terms)
    # U, V, W are 2x2 matrices flattened to 4-dim vectors.
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # We define the 7 Strassen products for 2x2x2.
    # Each product is a scalar multiplication of linear forms.
    # But we need the rank-1 tensor decomposition.
    # The 7 terms for 2x2x2 are:
    # 1. (a+d)(e+h) -> U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # 2. (c+d)e -> U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
    # 3. a(f-h) -> U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1]
    # 4. d(g-e) -> U=[0,0,0,1], V=[-1,0,1,0], W=[1,0,1,0]
    # 5. (a+b)h -> U=[1,1,0,0], V=[0,0,0,1], W=[-1,0,0,1]
    # 6. (c-a)(e+f) -> U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    # 7. (b-d)(g+h) -> U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
    
    # Note: The W indices above are for the local 2x2 output C.
    # C[0,0] gets from 1,4,5,7?
    # Let's verify W for term 1: W=[1,0,0,1] means W[0,0]=1, W[1,1]=1.
    # Term 1: (a+d)(e+h).
    # C00 = M1 + M4 - M5 + M7.
    # M1 contributes +1 to C00.
    # C11 = M1 - M2 + M3 + M6.
    # M1 contributes +1 to C11.
    # So W1 has 1 at (0,0) and (1,1). Correct.
    
    strassen_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,0,1]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0])
    ]
    
    # Now, 4x4x4 is 7 Strassen products of 2x2x2.
    # Each product M_k involves linear combinations of A and B blocks.
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    # M1 = (A00+A11)(B00+B11)
    # A00: (0,0)-(1,1). A11: (2,2)-(3,3).
    # Local (i,j) in A00 -> Global (i,j).
    # Local (i,j) in A11 -> Global (i+2, j+2).
    # So for local index li (0..3), global row is li//2 + 2*(li//2)? No.
    # Let's define mapping for A blocks.
    # Block (bi, bj) where bi, bj in {0,1}.
    # Global row = bi*2 + i_local. Global col = bj*2 + j_local.
    
    # For each of the 7 outer Strassen products, we define:
    # A_map: list of 4 (bi, bj, coeff) for the 4 local positions?
    # Actually, the linear form is a sum of entries from A.
    # For M1, A_part = A00 + A11.
    # Local (0,0) of A_part is A[0,0] + A[2,2].
    # Local (0,1) of A_part is A[0,1] + A[2,3].
    # Local (1,0) of A_part is A[1,0] + A[3,2].
    # Local (1,1) of A_part is A[1,1] + A[3,3].
    
    # So for a local index li (0..3) in the 2x2 A_part,
    # it corresponds to a sum of global A entries.
    
    # Let's define the 7 outer products with their A and B block mappings.
    # A_blocks: dict local_li -> list of (global_row, global_col, coeff)
    
    def get_block_map(blocks):
        # blocks is list of (bi, bj, coeff)
        # returns dict li -> list of (gi, gj, c)
        m = {}
        for bi, bj, c in blocks:
            for i in range(2):
                for j in range(2):
                    li = i*2 + j
                    gi = bi*2 + i
                    gj = bj*2 + j
                    if li not in m:
                        m[li] = []
                    m[li].append((gi, gj, c))
        return m

    # M1: A00 + A11
    A1_map = get_block_map([(0,0,1), (1,1,1)])
    B1_map = get_block_map([(0,0,1), (1,1,1)])
    C1_contrib = [(0,0,1), (1,1,1)] # C00 +1, C11 +1
    
    # M2: A10 + A11
    A2_map = get_block_map([(1,0,1), (1,1,1)])
    B2_map = get_block_map([(0,0,1)])
    C2_contrib = [(1,0,1), (1,1,-1)]
    
    # M3: A00
    A3_map = get_block_map([(0,0,1)])
    B3_map = get_block_map([(0,1,1), (1,1,-1)])
    C3_contrib = [(0,1,1), (1,1,1)]
    
    # M4: A11
    A4_map = get_block_map([(1,1,1)])
    B4_map = get_block_map([(1,0,1), (0,0,-1)])
    C4_contrib = [(0,0,1), (1,0,1)]
    
    # M5: A00 + A01
    A5_map = get_block_map([(0,0,1), (0,1,1)])
    B5_map = get_block_map([(1,1,1)])
    C5_contrib = [(0,0,-1), (0,1,1)]
    
    # M6: A10 - A00
    A6_map = get_block_map([(1,0,1), (0,0,-1)])
    B6_map = get_block_map([(0,0,1), (0,1,1)])
    C6_contrib = [(1,1,1)]
    
    # M7: A01 - A11
    A7_map = get_block_map([(0,1,1), (1,1,-1)])
    B7_map = get_block_map([(1,0,1), (1,1,1)])
    C7_contrib = [(0,0,1)]
    
    outer_products = [
        (A1_map, B1_map, C1_contrib),
        (A2_map, B2_map, C2_contrib),
        (A3_map, B3_map, C3_contrib),
        (A4_map, B4_map, C4_contrib),
        (A5_map, B5_map, C5_contrib),
        (A6_map, B6_map, C6_contrib),
        (A7_map, B7_map, C7_contrib)
    ]
    
    for A_map, B_map, C_contrib in outer_products:
        for u_loc, v_loc, w_loc in strassen_2x2:
            # u_loc, v_loc, w_loc are 4-dim lists (flattened 2x2)
            # For each local index, add contributions to global U, V, W
            for li in range(4):
                # U contribution
                if u_loc[li] != 0:
                    for gi, gj, c in A_map[li]:
                        U[gi*4 + gj][col] += u_loc[li] * c
                
                # V contribution (li is j*2+k? No, li is local index in V)
                # V local index is j*2+k.
                # B_map keys are local indices j*2+k.
                if v_loc[li] != 0:
                    for gj, gk, c in B_map[li]:
                        V[gj*5 + gk][col] += v_loc[li] * c
                
                # W contribution
                # w_loc[li] where li is k*2+i?
                # W local index is k*2+i.
                # We need to add to global W for each C block.
                for bi, bj, sign in C_contrib:
                    # Local W index li = k_local * 2 + i_local
                    k_local = li // 2
                    i_local = li % 2
                    gk = bj * 2 + k_local
                    gi = bi * 2 + i_local
                    W[gk*4 + gi][col] += w_loc[li] * sign
            
            col += 1
    
    # Now add 4x4x1 part (rank 16)
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
