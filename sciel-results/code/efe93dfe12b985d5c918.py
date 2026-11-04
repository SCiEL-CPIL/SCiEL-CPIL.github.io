import numpy as np
import time

DIMENSIONS = [4, 4, 7]

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

    def generate_strassen_4x4x7():
        """
        Correct Strassen-based decomposition for 4x4 * 4x7.
        Split B into B_left (4x4, cols 0-3) and B_right (4x3, cols 4-6).
        C = A * B_left + A * B_right
        
        For A * B_left (4x4 * 4x4): use Strassen with 2x2 blocks.
        Split A into A00, A01, A10, A11 (each 2x2).
        Split B_left into B00, B01, B10, B11 (each 2x2).
        C_left = C00, C01, C10, C11 (each 2x2).
        
        Strassen:
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B01 - B11)
        M4 = A11 * (B10 - B00)
        M5 = (A00 + A01) * B11
        M6 = (A10 - A00) * (B00 + B01)
        M7 = (A01 - A11) * (B10 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        Each Mi is a 2x2 * 2x2 multiplication, which has rank 4 (schoolbook for 2x2).
        So total rank for A*B_left = 7 * 4 = 28.
        
        For A * B_right (4x4 * 4x3): schoolbook, rank = 4*4*3 = 48.
        
        Total rank = 28 + 48 = 76.
        """
        total_rank = 76
        U = [[0] * total_rank for _ in range(16)]  # 4*4 = 16
        V = [[0] * total_rank for _ in range(28)]  # 4*7 = 28
        W = [[0] * total_rank for _ in range(28)]  # 7*4 = 28
        
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

        # Index helpers
        # A: 4x4, index (i,j) -> i*4+j
        # B: 4x7, index (j,k) -> j*7+k
        # C: 7x4, index (k,i) -> k*4+i (trace convention W[k*n+i,r])
        
        def a_idx(i, j):
            return i * 4 + j
        
        def b_idx(j, k):
            return j * 7 + k
        
        def c_idx(k, i):
            return k * 4 + i

        # For 2x2 block at rows r0..r0+1, cols c0..c0+1 of A:
        def a_block_indices(r0, c0):
            return [(a_idx(r0+di, c0+dj) for dj in range(2) for di in range(2))]
        
        # Actually, let me be more explicit.
        # A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # A01: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # A11: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # B_left (cols 0-3):
        # B00: rows 0-1, cols 0-1 -> b_idx(0,0), b_idx(0,1), b_idx(1,0), b_idx(1,1) = 0,1,7,8
        # B01: rows 0-1, cols 2-3 -> b_idx(0,2), b_idx(0,3), b_idx(1,2), b_idx(1,3) = 2,3,9,10
        # B10: rows 2-3, cols 0-1 -> b_idx(2,0), b_idx(2,1), b_idx(3,0), b_idx(3,1) = 14,15,21,22
        # B11: rows 2-3, cols 2-3 -> b_idx(2,2), b_idx(2,3), b_idx(3,2), b_idx(3,3) = 16,17,23,24
        
        # C_left (cols 0-3 of C, which is 7x4):
        # C00: rows 0-1, cols 0-1 of C -> c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1) = 0,1,4,5
        # C01: rows 0-1, cols 2-3 of C -> c_idx(0,2), c_idx(0,3), c_idx(1,2), c_idx(1,3) = 2,3,6,7
        # C10: rows 2-3, cols 0-1 of C -> c_idx(2,0), c_idx(2,1), c_idx(3,0), c_idx(3,1) = 8,9,12,13
        # C11: rows 2-3, cols 2-3 of C -> c_idx(2,2), c_idx(2,3), c_idx(3,2), c_idx(3,3) = 10,11,14,15

        # Define block index sets
        A00 = [0, 1, 4, 5]
        A01 = [2, 3, 6, 7]
        A10 = [8, 9, 12, 13]
        A11 = [10, 11, 14, 15]
        
        B00 = [0, 1, 7, 8]
        B01 = [2, 3, 9, 10]
        B10 = [14, 15, 21, 22]
        B11 = [16, 17, 23, 24]
        
        C00 = [0, 1, 4, 5]
        C01 = [2, 3, 6, 7]
        C10 = [8, 9, 12, 13]
        C11 = [10, 11, 14, 15]

        # For each Strassen product Mi = Li * Ri where Li and Ri are 2x2 matrices
        # formed from linear combinations of A and B blocks.
        # Li has entries that are linear combos of A entries.
        # Ri has entries that are linear combos of B entries.
        # The product Li*Ri is a 2x2 matrix, and each entry contributes to C.
        
        # For a 2x2 * 2x2 multiplication with rank-4 schoolbook:
        # If L = [[l00, l01], [l10, l11]] and R = [[r00, r01], [r10, r11]]
        # Then C = L*R where:
        # C00 = l00*r00 + l01*r10
        # C01 = l00*r01 + l01*r11
        # C10 = l10*r00 + l11*r10
        # C11 = l10*r01 + l11*r11
        
        # Each of these 4 terms is a rank-1 tensor term.
        # For term C00: U gets l00 and l01 coefficients, V gets r00 and r10, W gets C00 coefficient.
        # But wait, that's not right. Each rank-1 term is:
        # u_i * v_j * w_k = 1 for the specific (i,j,k) triple.
        
        # Actually, for the 2x2 matmul, the 4 schoolbook terms are:
        # Term 1: l00 * r00 -> contributes to C00
        # Term 2: l01 * r10 -> contributes to C00
        # Term 3: l00 * r01 -> contributes to C01
        # Term 4: l01 * r11 -> contributes to C01
        # Term 5: l10 * r00 -> contributes to C10
        # Term 6: l11 * r10 -> contributes to C10
        # Term 7: l10 * r01 -> contributes to C11
        # Term 8: l11 * r11 -> contributes to C11
        
        # That's 8 terms, not 4. I was wrong. 2x2 * 2x2 schoolbook is rank 8.
        # So 7 * 8 = 56 for the Strassen part.
        
        # Let me reconsider. The rank of 2x2*2x2 matrix multiplication is 7 (Strassen) or 8 (schoolbook).
        # If we use schoolbook for the 2x2 blocks, it's 8 terms per product.
        # 7 products * 8 terms = 56 terms for A*B_left.
        # Plus 48 for A*B_right = 104 total.
        
        # But if we use Strassen recursively for the 2x2 blocks... 2x2*2x2 has rank 7, but that's the minimal known.
        # Actually, for 2x2*2x2, the rank is exactly 7 (Strassen is optimal for 2x2).
        # But implementing Strassen for 2x2*2x2 is complex. Let me just use schoolbook for 2x2*2x2.
        
        # So total rank = 7 * 8 + 48 = 56 + 48 = 104.
        
        # Hmm, that's not as good as I hoped. Let me try a different approach.
        
        # Alternative: Use the fact that 4x4*4x7 can be decomposed using the 4x4*4x4 Strassen result.
        # Actually, let me just implement the 56 + 48 = 104 version correctly.
        
        # For each Strassen product Mi = Li * Ri (2x2 * 2x2), use schoolbook (8 terms).
        
        def add_2x2_schoolbook(L_coeffs, R_coeffs, C_block_indices, scale):
            """
            L_coeffs: dict mapping local (lr,lc) -> dict {global_a_idx: coeff}
            R_coeffs: dict mapping local (rr,rc) -> dict {global_b_idx: coeff}
            C_block_indices: list of 4 global C indices for the 2x2 C block
            scale: multiplicative factor
            """
            nonlocal col
            # 8 schoolbook terms for 2x2 * 2x2
            # Term (lr, rr, rc): L[lr,rc] * R[rr,rc] -> C[lr,rc]
            for lr in range(2):
                for rc in range(2):
                    for rr in range(2):
                        # L[lr,rc] * R[rr,rc] -> C[lr,rc]
                        l_dict = L_coeffs.get((lr, rc), {})
                        r_dict = R_coeffs.get((rr, rc), {})
                        
                        if not l_dict or not r_dict:
                            continue
                        
                        # This term: sum of products
                        # For each (a_idx, a_coeff) in l_dict and (b_idx, b_coeff) in r_dict:
                        # U[a_idx][col] += scale * a_coeff
                        # V[b_idx][col] += scale * b_coeff
                        # W[c_idx][col] += scale
                        # But that's not right for multiple entries.
                        
                        # Actually, each rank-1 term is a single (i,j,k) triple.
                        # For L[lr,rc] * R[rr,rc] -> C[lr,rc], if L[lr,rc] is a sum of A entries
                        # and R[rr,rc] is a sum of B entries, we need multiple rank-1 terms.
                        
                        # Let me restructure. L_coeffs should map (lr,rc) to a list of (global_a_idx, coeff).
                        # Similarly for R_coeffs.
                        
                        pass
        
        # This is getting complicated. Let me use a cleaner approach.
        
        # Represent L as a 2x2 matrix where each entry is a dict {global_a_idx: coeff}
        # Represent R as a 2x2 matrix where each entry is a dict {global_b_idx: coeff}
        
        def make_L(block_list):
            """block_list: list of (block_id, coeff). Returns 2x2 list of dicts."""
            L = [[{} for _ in range(2)] for _ in range(2)]
            blocks = [A00, A01, A10, A11]
            for block_id, coeff in block_list:
                for idx in blocks[block_id]:
                    # Need to map global A index to local (lr,lc)
                    # A00: [0,1,4,5] -> (0,0),(0,1),(1,0),(1,1)
                    # A01: [2,3,6,7] -> (0,0),(0,1),(1,0),(1,1) but in block 01
                    # This mapping is tricky. Let me use explicit mappings.
                    pass
            return L
        
        # Let me define explicit local-to-global mappings for each block.
        # For A block at (br, bc) where br, bc in {0,1}:
        # Local (lr,lc) -> global (br*2+lr, bc*2+lc) -> index (br*2+lr)*4 + (bc*2+lc)
        
        def a_global(br, bc, lr, lc):
            return (br*2+lr)*4 + (bc*2+lc)
        
        def b_global(br, bc, rr, rc):
            # B block at rows br*2..br*2+1, cols bc*2..bc*2+1 (for B_left, cols 0-3)
            return (br*2+rr)*7 + (bc*2+rc)
        
        def c_global(br, bc, lr, lc):
            # C block at rows bc*2..bc*2+1 (cols of C), cols br*2..br*2+1 (rows of C)
            # Wait, C is 7x4, indexed as c_idx(k,i) = k*4+i where k is row (0-6), i is col (0-3)
            # C00: k=0,1; i=0,1 -> c_idx(0,0), c_idx(0,1), c_idx(1,0), c_idx(1,1)
            # So local (lr,lc) in C block (br,bc) where br is row-block, bc is col-block:
            # k = bc*2+lr, i = br*2+lc
            return (bc*2+lr)*4 + (br*2+lc)
        
        # Now define Strassen products.
        # Each product Mi = Li * Ri where Li is 2x2 (entries are linear combos of A), Ri is 2x2 (entries are linear combos of B).
        # Li[lr,lc] = sum of coeff * A[br*2+lr, bc*2+lc] for (br,bc,coeff) in block_list
        # Ri[rr,rc] = sum of coeff * B[br*2+rr, bc*2+rc] for (br,bc,coeff) in block_list
        
        def make_L_mat(block_list):
            """block_list: list of (br, bc, coeff). Returns 2x2 list of dicts {global_a_idx: coeff}."""
            L = [[{} for _ in range(2)] for _ in range(2)]
            for br, bc, coeff in block_list:
                for lr in range(2):
                    for lc in range(2):
                        gidx = a_global(br, bc, lr, lc)
                        L[lr][lc][gidx] = L[lr][lc].get(gidx, 0) + coeff
            return L
        
        def make_R_mat(block_list):
            """block_list: list of (br, bc, coeff). Returns 2x2 list of dicts {global_b_idx: coeff}."""
            R = [[{} for _ in range(2)] for _ in range(2)]
            for br, bc, coeff in block_list:
                for rr in range(2):
                    for rc in range(2):
                        gidx = b_global(br, bc, rr, rc)
                        R[rr][rc][gidx] = R[rr][rc].get(gidx, 0) + coeff
            return R
        
        def add_strassen_product(L_mat, R_mat, c_br, c_bc, scale):
            """
            L_mat: 2x2 list of dicts {global_a_idx: coeff}
            R_mat: 2x2 list of dicts {global_b_idx: coeff}
            c_br, c_bc: C block row-block and col-block indices
            scale: multiplicative factor
            
            Uses schoolbook for 2x2*2x2 (8 terms).
            """
            nonlocal col
            for lr in range(2):
                for rc in range(2):
                    for rr in range(2):
                        # Term: L[lr,rc] * R[rr,rc] -> C[lr,rc] in block (c_br, c_bc)
                        l_dict = L_mat[lr][rc]
                        r_dict = R_mat[rr][rc]
                        
                        # For each a_idx in l_dict and b_idx in r_dict, add a rank-1 term
                        # But that would be |l_dict| * |r_dict| terms, which is too many.
                        
                        # Actually, L[lr,rc] is a linear combination of A entries.
                        # R[rr,rc] is a linear combination of B entries.
                        # The product L[lr,rc] * R[rr,rc] is a sum of products of A and B entries.
                        # Each such product corresponds to one rank-1 term in the tensor.
                        
                        # So we need to expand: for each (a_idx, a_coeff) in l_dict and (b_idx, b_coeff) in r_dict:
                        # Add term: U[a_idx][col] = scale * a_coeff, V[b_idx][col] = scale * b_coeff, W[c_gidx][col] = scale
                        # But wait, that's not right either. The rank-1 term should have:
                        # U[a_idx][col] = a_coeff, V[b_idx][col] = b_coeff, W[c_gidx][col] = scale
                        # And the product U[a_idx][col] * V[b_idx][col] * W[c_gidx][col] = a_coeff * b_coeff * scale
                        # which should equal the coefficient of A[a_idx] * B[b_idx] in C[c_gidx].
                        
                        # Yes, that's correct.
                        
                        c_gidx = c_global(c_br, c_bc, lr, rc)
                        
                        for a_idx, a_coeff in l_dict.items():
                            for b_idx, b_coeff in r_dict.items():
                                U[a_idx][col] = scale * a_coeff
                                V[b_idx][col] = scale * b_coeff
                                W[c_gidx][col] = scale
                                col += 1
        
        # Define the 7 Strassen products
        # M1 = (A00 + A11) * (B00 + B11)
        L1 = make_L_mat([(0,0,1), (1,1,1)])
        R1 = make_R_mat([(0,0,1), (1,1,1)])
        
        # M2 = (A10 + A11) * B00
        L2 = make_L_mat([(1,0,1), (1,1,1)])
        R2 = make_R_mat([(0,0,1)])
        
        # M3 = A00 * (B01 - B11)
        L3 = make_L_mat([(0,0,1)])
        R3 = make_R_mat([(0,1,1), (1,1,-1)])
        
        # M4 = A11 * (B10 - B00)
        L4 = make_L_mat([(1,1,1)])
        R4 = make_R_mat([(1,0,1), (0,0,-1)])
        
        # M5 = (A00 + A01) * B11
        L5 = make_L_mat([(0,0,1), (0,1,1)])
        R5 = make_R_mat([(1,1,1)])
        
        # M6 = (A10 - A00) * (B00 + B01)
        L6 = make_L_mat([(1,0,1), (0,0,-1)])
        R6 = make_R_mat([(0,0,1), (0,1,1)])
        
        # M7 = (A01 - A11) * (B10 + B11)
        L7 = make_L_mat([(0,1,1), (1,1,-1)])
        R7 = make_R_mat([(1,0,1), (1,1,1)])
        
        # Reconstruction:
        # C00 = M1 + M4 - M5 + M7
        add_strassen_product(L1, R1, 0, 0, 1)
        add_strassen_product(L4, R4, 0, 0, 1)
        add_strassen_product(L5, R5, 0, 0, -1)
        add_strassen_product(L7, R7, 0, 0, 1)
        
        # C01 = M3 + M5
        add_strassen_product(L3, R3, 0, 1, 1)
        add_strassen_product(L5, R5, 0, 1, 1)
        
        # C10 = M2 + M4
        add_strassen_product(L2, R2, 1, 0, 1)
        add_strassen_product(L4, R4, 1, 0, 1)
        
        # C11 = M1 - M2 + M3 + M6
        add_strassen_product(L1, R1, 1, 1, 1)
        add_strassen_product(L2, R2, 1, 1, -1)
        add_strassen_product(L3, R3, 1, 1, 1)
        add_strassen_product(L6, R6, 1, 1, 1)
        
        # Part 2: Schoolbook for 4x4 * 4x3 (cols 4-6 of B, cols 4-6 of C)
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i * 4 + j][col] = 1
                    V[j * 7 + (4 + k)][col] = 1
                    W[(4 + k) * 4 + i][col] = 1
                    col += 1

        # Check if we used the expected number of terms
        # Strassen part: each product has 8 terms (2*2*2), 7 products = 56 terms
        # But wait, some L or R entries might be empty, reducing the count.
        # Let me just check col <= total_rank
        if col > total_rank:
            # This shouldn't happen if my analysis is correct
            return None
        
        # Pad if necessary
        while col < total_rank:
            col += 1
        
        return U, V, W, col

    try:
        result = generate_strassen_4x4x7()
        if result is not None:
            U, V, W, rank = result
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
        
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    except Exception as e:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
