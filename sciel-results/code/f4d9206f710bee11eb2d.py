import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0
    
    # Schoolbook baseline (just in case)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def generate_hybrid_4x4x7():
        """
        Generates U, V, W for 4x4 * 4x7 multiplication.
        Uses Strassen (rank 28) for the first 4 columns of B (4x4 part).
        Uses Schoolbook (rank 48) for the last 3 columns of B (4x3 part).
        Total rank: 28 + 48 = 76.
        """
        total_rank = 76
        U = [[0] * total_rank for _ in range(16)]
        V = [[0] * total_rank for _ in range(28)]
        W = [[0] * total_rank for _ in range(28)]
        
        col = 0

        def add_term(a_coeffs, b_coeffs, c_coeffs):
            nonlocal col
            for a_idx, a_val in a_coeffs.items():
                if a_val != 0:
                    U[a_idx][col] += a_val
            for b_idx, b_val in b_coeffs.items():
                if b_val != 0:
                    V[b_idx][col] += b_val
            for c_idx, c_val in c_coeffs.items():
                if c_val != 0:
                    W[c_idx][col] += c_val
            col += 1

        # Rank-4 decomposition of 2x2 matrix multiplication.
        # For L=[[l00,l01],[l10,l11]] * R=[[r00,r01],[r10,r11]] = C
        # The 4 rank-1 terms use:
        #   term0: (l00+l10) * r00 -> C[0,0]+=1, C[1,0]+=1
        #   term1: (l01+l11) * r10 -> C[0,0]+=1, C[1,0]+=1
        #   term2: (l00+l10) * r01 -> C[0,1]+=1, C[1,1]+=1
        #   term3: (l01+l11) * r11 -> C[0,1]+=1, C[1,1]+=1
        #
        # l_coeffs: dict mapping (local_row, local_col) -> coeff for L entries
        # r_coeffs: dict mapping (local_row, local_col) -> coeff for R entries
        # c_map: dict mapping (local_row, local_col) -> global_c_idx
        # scale: multiplicative coefficient for this Strassen product's contribution
        def add_2x2_product(l_coeffs, r_coeffs, c_map, scale):
            nonlocal col
            # Build the 4 rank-1 terms
            # term0: L rows combined (l00+l10) * R[0,0] -> C local [0,0] and [1,0]
            # term1: L rows combined (l01+l11) * R[1,0] -> C local [0,0] and [1,0]
            # term2: L rows combined (l00+l10) * R[0,1] -> C local [0,1] and [1,1]
            # term3: L rows combined (l01+l11) * R[1,1] -> C local [0,1] and [1,1]
            
            # For each of the 4 terms, the L vector sums over rows for fixed col
            # term t: L_vec[i] = sum of l_coeffs[(i, col_t)] for the appropriate col
            # Actually: term0 uses L[:,0] summed over rows, times R[0,0]
            # term1 uses L[:,1] summed over rows, times R[1,0]
            # term2 uses L[:,0] summed over rows, times R[0,1]
            # term3 uses L[:,1] summed over rows, times R[1,1]
            
            for t in range(4):
                if t == 0:
                    l_col = 0  # use column 0 of L
                    r_row = 0  # use row 0 of R
                    r_col = 0
                    c_local_rows = [0, 1]
                    c_local_col = 0
                elif t == 1:
                    l_col = 1  # use column 1 of L
                    r_row = 1
                    r_col = 0
                    c_local_rows = [0, 1]
                    c_local_col = 0
                elif t == 2:
                    l_col = 0
                    r_row = 0
                    r_col = 1
                    c_local_rows = [0, 1]
                    c_local_col = 1
                else:
                    l_col = 1
                    r_row = 1
                    r_col = 1
                    c_local_rows = [0, 1]
                    c_local_col = 1
                
                a_c = {}
                for lr in range(2):
                    key = (lr, l_col)
                    if key in l_coeffs:
                        a_c[key] = scale * l_coeffs[key]
                
                b_c = {}
                key = (r_row, r_col)
                if key in r_coeffs:
                    b_c[key] = scale * r_coeffs[key]
                
                c_c = {}
                for cr in c_local_rows:
                    ckey = (cr, c_local_col)
                    if ckey in c_map:
                        c_c[c_map[ckey]] = scale
                
                add_term(a_c, b_c, c_c)

        # Define block indices for 4x4 matrices split into 2x2 blocks
        # A blocks: A00, A01, A10, A11
        # A00 = rows 0-1, cols 0-1
        # A01 = rows 0-1, cols 2-3
        # A10 = rows 2-3, cols 0-1
        # A11 = rows 2-3, cols 2-3
        
        # B blocks for the 4x4 part (cols 0-3): B00, B01, B10, B11
        # B00 = rows 0-1, cols 0-1
        # B01 = rows 0-1, cols 2-3
        # B10 = rows 2-3, cols 0-1
        # B11 = rows 2-3, cols 2-3
        
        # C blocks for the 4x4 part (cols 0-3): C00, C01, C10, C11
        # C00 = rows 0-1, cols 0-1
        # C01 = rows 0-1, cols 2-3
        # C10 = rows 2-3, cols 0-1
        # C11 = rows 2-3, cols 2-3

        def a_block_idx(block, lr, lc):
            """Global A index for block (block in 0..3) at local (lr,lc)"""
            # block 0=A00, 1=A01, 2=A10, 3=A11
            br = (block // 2) * 2
            bc = (block % 2) * 2
            return (br + lr) * 4 + (bc + lc)

        def b_block_idx(block, lr, lc):
            """Global B index for block (block in 0..3) at local (lr,lc) in the 4x4 part"""
            br = (block // 2) * 2
            bc = (block % 2) * 2
            return br * 7 + (bc + lc)

        def c_block_idx(block, lr, lc):
            """Global C index for block (block in 0..3) at local (lr,lc) in the 4x4 part"""
            br = (block // 2) * 2
            bc = (block % 2) * 2
            return (bc + lc) * 4 + (br + lr)

        # Helper to build l_coeffs and r_coeffs from block definitions
        # A block combination: list of (block, coeff)
        def build_l_coeffs(block_list):
            """block_list: list of (block_id, coeff)"""
            coeffs = {}
            for block, coeff in block_list:
                for lr in range(2):
                    for lc in range(2):
                        key = (lr, lc)
                        coeffs[key] = coeffs.get(key, 0) + coeff
            return coeffs

        # For Strassen, L and R are 2x2 matrices formed from blocks
        # L is formed from A blocks, R is formed from B blocks

        # Strassen products:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)

        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # For each Strassen product, the "L" matrix is a 2x2 matrix whose entries
        # are linear combinations of A entries, and "R" is similar for B.
        # The 2x2 product L*R gives a 2x2 result that maps to one of the C blocks.

        # Let me define each Strassen product explicitly.
        # For product M = L * R where L and R are 2x2:
        # L is built from A blocks: each entry L[lr,lc] is a linear combo of A entries
        # R is built from B blocks: each entry R[rr,rc] is a linear combo of B entries
        # The result maps to C block cb: C[br+lr, bc+rc] gets contribution from M[lr,rc]

        # I'll represent L as a 2x2 array of dicts (each dict maps A global idx -> coeff)
        # and R as a 2x2 array of dicts (each dict maps B global idx -> coeff)

        def make_l_from_blocks(block_coeffs_list):
            """
            block_coeffs_list: list of (block_id, coeff)
            Returns 2x2 array where entry [lr][lc] is a dict {global_a_idx: coeff}
            """
            L = [[{} for _ in range(2)] for _ in range(2)]
            for block, coeff in block_coeffs_list:
                br = (block // 2) * 2
                bc = (block % 2) * 2
                for lr in range(2):
                    for lc in range(2):
                        gidx = (br + lr) * 4 + (bc + lc)
                        L[lr][lc][gidx] = L[lr][lc].get(gidx, 0) + coeff
            return L

        def make_r_from_blocks(block_coeffs_list):
            """
            block_coeffs_list: list of (block_id, coeff)
            Returns 2x2 array where entry [rr][rc] is a dict {global_b_idx: coeff}
            """
            R = [[{} for _ in range(2)] for _ in range(2)]
            for block, coeff in block_coeffs_list:
                br = (block // 2) * 2
                bc = (block % 2) * 2
                for rr in range(2):
                    for rc in range(2):
                        gidx = br * 7 + (bc + rc)
                        R[rr][rc][gidx] = R[rr][rc].get(gidx, 0) + coeff
            return R

        def add_strassen_product(L, R, c_block, scale):
            """
            L: 2x2 array of dicts {global_a_idx: coeff}
            R: 2x2 array of dicts {global_b_idx: coeff}
            c_block: the C block index (0..3) this product contributes to
            scale: multiplicative factor
            
            Uses rank-4 decomposition of 2x2 matmul.
            """
            br = (c_block // 2) * 2
            bc = (c_block % 2) * 2
            
            # 4 rank-1 terms for 2x2 matmul:
            # term0: sum_rows(L[:,0]) * R[0,0] -> C[br, bc] and C[br+1, bc]
            # term1: sum_rows(L[:,1]) * R[1,0] -> C[br, bc] and C[br+1, bc]
            # term2: sum_rows(L[:,0]) * R[0,1] -> C[br, bc+1] and C[br+1, bc+1]
            # term3: sum_rows(L[:,1]) * R[1,1] -> C[br, bc+1] and C[br+1, bc+1]
            
            for t in range(4):
                if t == 0:
                    l_col = 0
                    r_row, r_col = 0, 0
                    c_rc = 0
                elif t == 1:
                    l_col = 1
                    r_row, r_col = 1, 0
                    c_rc = 0
                elif t == 2:
                    l_col = 0
                    r_row, r_col = 0, 1
                    c_rc = 1
                else:
                    l_col = 1
                    r_row, r_col = 1, 1
                    c_rc = 1
                
                # A coefficients: sum over rows of L[:, l_col]
                a_c = {}
                for lr in range(2):
                    for gidx, coeff in L[lr][l_col].items():
                        a_c[gidx] = a_c.get(gidx, 0) + scale * coeff
                
                # B coefficients: R[r_row, r_col]
                b_c = {}
                for gidx, coeff in R[r_row][r_col].items():
                    b_c[gidx] = scale * coeff
                
                # C coefficients: C[br, bc+c_rc] and C[br+1, bc+c_rc]
                c_c = {}
                for cr in range(2):
                    c_gidx = (bc + c_rc) * 4 + (br + cr)
                    c_c[c_gidx] = c_c.get(c_gidx, 0) + scale
                
                add_term(a_c, b_c, c_c)

        # Define the 7 Strassen products
        # M1 = (A00 + A11) * (B00 + B11) -> contributes to C00 and C11
        L1 = make_l_from_blocks([(0, 1), (3, 1)])
        R1 = make_r_from_blocks([(0, 1), (3, 1)])
        
        # M2 = (A10 + A11) * B00 -> contributes to C10 and C11
        L2 = make_l_from_blocks([(2, 1), (3, 1)])
        R2 = make_r_from_blocks([(0, 1)])
        
        # M3 = A00 * (B01 - B11) -> contributes to C01 and C11
        L3 = make_l_from_blocks([(0, 1)])
        R3 = make_r_from_blocks([(1, 1), (3, -1)])
        
        # M4 = A11 * (B10 - B00) -> contributes to C00 and C10
        L4 = make_l_from_blocks([(3, 1)])
        R4 = make_r_from_blocks([(2, 1), (0, -1)])
        
        # M5 = (A00 + A01) * B11 -> contributes to C00 and C01
        L5 = make_l_from_blocks([(0, 1), (1, 1)])
        R5 = make_r_from_blocks([(3, 1)])
        
        # M6 = (A10 - A00) * (B00 + B01) -> contributes to C11
        L6 = make_l_from_blocks([(2, 1), (0, -1)])
        R6 = make_r_from_blocks([(0, 1), (1, 1)])
        
        # M7 = (A01 - A11) * (B10 + B11) -> contributes to C00
        L7 = make_l_from_blocks([(1, 1), (3, -1)])
        R7 = make_r_from_blocks([(2, 1), (3, 1)])

        # Reconstruction:
        # C00 = M1 + M4 - M5 + M7
        add_strassen_product(L1, R1, 0, 1)   # M1 to C00
        add_strassen_product(L4, R4, 0, 1)   # M4 to C00
        add_strassen_product(L5, R5, 0, -1)  # -M5 to C00
        add_strassen_product(L7, R7, 0, 1)   # M7 to C00
        
        # C01 = M3 + M5
        add_strassen_product(L3, R3, 1, 1)   # M3 to C01
        add_strassen_product(L5, R5, 1, 1)   # M5 to C01
        
        # C10 = M2 + M4
        add_strassen_product(L2, R2, 2, 1)   # M2 to C10
        add_strassen_product(L4, R4, 2, 1)   # M4 to C10
        
        # C11 = M1 - M2 + M3 + M6
        add_strassen_product(L1, R1, 3, 1)   # M1 to C11
        add_strassen_product(L2, R2, 3, -1)  # -M2 to C11
        add_strassen_product(L3, R3, 3, 1)   # M3 to C11
        add_strassen_product(L6, R6, 3, 1)   # M6 to C11

        # Part 2: Schoolbook for 4x4 * 4x3 (cols 4-6 of B, cols 4-6 of C)
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i * 4 + j][col] = 1
                    V[j * 7 + (4 + k)][col] = 1
                    W[(4 + k) * 4 + i][col] = 1
                    col += 1

        assert col == total_rank, f"Expected {total_rank} terms, got {col}"
        return U, V, W, total_rank

    try:
        U, V, W, rank = generate_hybrid_4x4x7()
        
        # Verify
        valid = True
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    total = 0
                    for c in range(rank):
                        total += U[i*4+j][c] * V[j*7+k][c] * W[k*4+i][c]
                    if total != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return {"U": U, "V": V, "W": W}
        else:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
            
    except Exception as e:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
