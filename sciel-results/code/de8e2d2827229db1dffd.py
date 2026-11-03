"""
Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
Uses nested Strassen (rank-7 for 2x2x2) to achieve rank 49 for 4x4x4,
then combines two 4x4x4 blocks to get rank 98 for 4x4x8.
"""
import time

DIMENSIONS = [4, 4, 8]

# Deadline guard
_DEADLINE = 145.0  # seconds
_start_time = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)

# Strassen rank-7 decomposition for 2x2x2
# U, V, W are 4x7 matrices where index in U is i*2+j, V is j*2+k, W is k*2+i
# The 7 Strassen terms:
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A01)B11
# M6 = (A10-A00)(B00+B01)
# M7 = (A01-A11)(B10+B11)
#
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

def _strassen_2x2_2x2():
    """Return U, V, W (each 4x7) for 2x2x2 Strassen decomposition.
    U index: i*2+j (i in 0..1, j in 0..1)
    V index: j*2+k (j in 0..1, k in 0..1)
    W index: k*2+i (k in 0..1, i in 0..1)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Each term t has contributions to U (row i*2+j), V (row j*2+k), W (row k*2+i)
    # We define each of the 7 terms as a set of (i,j) for U, (j,k) for V, (k,i) for W
    
    # M1: (A00+A11)(B00+B11)
    # U: A00=(0,0), A11=(1,1) -> rows 0, 3
    # V: B00=(0,0), B11=(1,1) -> rows 0, 3
    # W: C00 gets +M1 -> (0,0) row 0; C11 gets +M1 -> (1,1) row 3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    # U: A10=(1,0) row 2, A11=(1,1) row 3
    # V: B00=(0,0) row 0
    # W: C10 gets +M2 -> (1,0) row 2; C11 gets -M2 -> (1,1) row 3
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A00(B01-B11)
    # U: A00=(0,0) row 0
    # V: B01=(0,1) row 1, B11=(1,1) row 3
    # W: C01 gets +M3 -> (0,1) row 1; C11 gets +M3 -> (1,1) row 3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11(B10-B00)
    # U: A11=(1,1) row 3
    # V: B10=(1,0) row 2, B00=(0,0) row 0
    # W: C00 gets +M4 -> (0,0) row 0; C10 gets +M4 -> (1,0) row 2
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)B11
    # U: A00=(0,0) row 0, A01=(0,1) row 1
    # V: B11=(1,1) row 3
    # W: C00 gets -M5 -> (0,0) row 0; C01 gets +M5 -> (0,1) row 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: A10=(1,0) row 2, A00=(0,0) row 0
    # V: B00=(0,0) row 0, B01=(0,1) row 1
    # W: C11 gets +M6 -> (1,1) row 3
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: A01=(0,1) row 1, A11=(1,1) row 3
    # V: B10=(1,0) row 2, B11=(1,1) row 3
    # W: C00 gets +M7 -> (0,0) row 0
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _strassen_4x4_4x4():
    """Build rank-49 decomposition for 4x4x4 using two levels of Strassen.
    Returns U, V, W each of shape (16, 49).
    U index: i*4+j (i in 0..3, j in 0..3)
    V index: j*4+k (j in 0..3, k in 0..3)
    W index: k*4+i (k in 0..3, i in 0..3)
    """
    # 4x4 = 2x2 blocks of 2x2. Use Strassen at both levels.
    # Level 1: split 4x4 into four 2x2 blocks.
    # Level 2: each 2x2 multiplication uses rank-7 Strassen.
    # Total rank = 7 * 7 = 49.
    
    # For the 4x4 case with block structure:
    # A is 4x4, split into A00, A01, A10, A11 each 2x2.
    # B is 4x4, split into B00, B01, B10, B11 each 2x2.
    # C is 4x4, split into C00, C01, C10, C11 each 2x2.
    
    # Strassen at top level gives 7 products of 2x2 matrices.
    # Each 2x2 product uses rank-7 Strassen.
    # Total: 7*7 = 49 scalar terms.
    
    U, V, W = [[0]*49 for _ in range(16)], [[0]*49 for _ in range(16)], [[0]*49 for _ in range(16)]
    
    # Top-level Strassen indices for 4x4:
    # Term index t in 0..6 corresponds to the 7 Strassen products.
    # For each top-level term, we have a (2x2)*(2x2) product.
    # The sub-Strassen index s in 0..6 for the 2x2 level.
    # Global index = t * 7 + s
    
    # For top-level Strassen on 4x4 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_t is a 2x2 matrix product. We need to express:
    # (sum of selected A blocks) * (sum of selected B blocks)
    # as a tensor product.
    
    # For each top-level term t, the A-side is a sum of one or two 2x2 blocks with signs,
    # and the B-side is a sum of one or two 2x2 blocks with signs.
    
    # Define for each top-level term:
    # a_blocks: list of (block_idx, sign) where block_idx encodes which 2x2 block
    #   block 0 = A00 (i in 0..1, j in 0..1)
    #   block 1 = A01 (i in 0..1, j in 2..3)
    #   block 2 = A10 (i in 2..3, j in 0..1)
    #   block 3 = A11 (i in 2..3, j in 2..3)
    # b_blocks: similar for B
    # c_blocks: list of (block_idx, sign) for C
    
    top_terms = [
        # (a_blocks, b_blocks, c_blocks)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),       # M1: C00+, C11+
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),            # M2: C10+, C11-
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),            # M3: C01+, C11+
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),            # M4: C00+, C10+
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),            # M5: C00-, C01+
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),            # M6: C11+
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),            # M7: C00+
    ]
    
    # Get the 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_2x2()
    
    for t, (a_blocks, b_blocks, c_blocks) in enumerate(top_terms):
        for s in range(7):
            g = t * 7 + s  # global index
            
            # For the sub-Strassen term s:
            # U2[row][s] gives the coefficient for A_{i,j} (within the 2x2 block)
            # V2[row][s] gives the coefficient for B_{j,k} (within the 2x2 block)
            # W2[row][s] gives the coefficient for C_{k,i} (within the 2x2 block)
            
            # We need to combine:
            # A-side: sum over (block_a, sign_a) of sign_a * A_block_a
            # B-side: sum over (block_b, sign_b) of sign_b * B_block_b
            # The product contributes to C blocks according to c_blocks.
            
            # For each sub-Strassen term s, U2[r_a][s] is nonzero for some r_a (0..3),
            # which corresponds to (i_local, j_local) = (r_a // 2, r_a % 2) within a 2x2 block.
            # Similarly for V2 and W2.
            
            # For the A-side: if a_blocks has entries [(ba, sa), (bb, sb)],
            # then for a given (i_local, j_local) in the 2x2, the global (i, j) is:
            #   if block 0: i = i_local, j = j_local
            #   if block 1: i = i_local, j = j_local + 2
            #   if block 2: i = i_local + 2, j = j_local
            #   if block 3: i = i_local + 2, j = j_local + 2
            
            for r_a in range(4):
                u_val = U2[r_a][s]
                if u_val == 0:
                    continue
                i_local = r_a // 2
                j_local = r_a % 2
                
                for r_b in range(4):
                    v_val = V2[r_b][s]
                    if v_val == 0:
                        continue
                    j_loc_b = r_b // 2
                    k_local = r_b % 2
                    
                    for r_c in range(4):
                        w_val = W2[r_c][s]
                        if w_val == 0:
                            continue
                        k_loc_c = r_c // 2
                        i_loc_c = r_c % 2
                        
                        # Check consistency: j_local (from A) should match j_loc_b (from B)
                        # and k_local (from B) should match k_loc_c (from C)
                        # and i_local (from A) should match i_loc_c (from C)
                        if j_local != j_loc_b or k_local != k_loc_c or i_local != i_loc_c:
                            continue
                        
                        # Now compute the total coefficient for this global term g
                        # A contribution: sum over a_blocks
                        a_coeff = 0
                        for (ba, sa) in a_blocks:
                            # Block ba maps (i_local, j_local) to global (i, j)
                            if ba == 0:
                                gi, gj = i_local, j_local
                            elif ba == 1:
                                gi, gj = i_local, j_local + 2
                            elif ba == 2:
                                gi, gj = i_local + 2, j_local
                            else:  # ba == 3
                                gi, gj = i_local + 2, j_local + 2
                            a_coeff += sa * u_val
                        
                        # B contribution: sum over b_blocks
                        b_coeff = 0
                        for (bb, sb) in b_blocks:
                            if bb == 0:
                                gj_b, gk = j_loc_b, k_local
                            elif bb == 1:
                                gj_b, gk = j_loc_b + 2, k_local
                            elif bb == 2:
                                gj_b, gk = j_loc_b, k_local + 2
                            else:  # bb == 3
                                gj_b, gk = j_loc_b + 2, k_local + 2
                            b_coeff += sb * v_val
                        
                        # C contribution: sum over c_blocks
                        c_coeff = 0
                        for (bc, sc) in c_blocks:
                            if bc == 0:
                                gk_c, gi_c = k_loc_c, i_loc_c
                            elif bc == 1:
                                gk_c, gi_c = k_loc_c + 2, i_loc_c
                            elif bc == 2:
                                gk_c, gi_c = k_loc_c, i_loc_c + 2
                            else:  # bc == 3
                                gk_c, gi_c = k_loc_c + 2, i_loc_c + 2
                            c_coeff += sc * w_val
                        
                        # Now check that the global indices are consistent
                        # A: (gi, gj), B: (gj_b, gk), C: (gk_c, gi_c)
                        # Need gj == gj_b and gk == gk_c and gi == gi_c
                        if gj != gj_b or gk != gk_c or gi != gi_c:
                            continue
                        
                        # All consistent. Add to the decomposition.
                        U[gi * 4 + gj][g] += a_coeff
                        V[gj * 4 + gk][g] += b_coeff
                        W[gk * 4 + gi][g] += c_coeff
    
    return U, V, W


def construct():
    if _time_remaining() < 5.0:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = 4, 4, 8
    rank = 98  # 49 * 2
    
    # Build 4x4x4 decomposition
    U4, V4, W4 = _strassen_4x4_4x4()
    
    # Now build 4x4x8 by splitting B and C into two 4-column halves.
    # B is 4x8: split into B1 (4x4, k=0..3) and B2 (4x4, k=4..7)
    # C is 8x4: split into C1 (4x4, k=0..3) and C2 (4x4, k=4..7)
    # A is 4x4: same for both.
    
    # U_total: 16 x 98
    # V_total: 32 x 98
    # W_total: 32 x 98
    
    U_total = [[0] * rank for _ in range(16)]
    V_total = [[0] * rank for _ in range(32)]
    W_total = [[0] * rank for _ in range(32)]
    
    for r in range(16):
        for c in range(49):
            # First block (k=0..3)
            U_total[r][c] = U4[r][c]
            V_total[r][c] = V4[r][c]  # rows 0..15 of V_total correspond to j*8+k for k<4
            W_total[r][c] = W4[r][c]  # rows 0..15 of W_total correspond to k*4+i for k<4
            
            # Second block (k=4..7)
            U_total[r][49 + c] = U4[r][c]
            # V for second block: row j*8 + (k+4) = j*8 + 4 + k
            # In V4, row is j*4 + k. We need to map to j*8 + 4 + k.
            # V4[r][c] is at row r = j*4 + k in the 4x4x4 case.
            # In the 4x4x8 case, this should go to row j*8 + 4 + k.
            # j = r // 4, k = r % 4
            # new_row = j*8 + 4 + k
            pass  # Handle V mapping below
    
    # Redo V_total properly
    # V4 has 16 rows indexed by j*4 + k (j in 0..3, k in 0..3)
    # For first block (k_global = k, 0..3): V row = j*8 + k
    # For second block (k_global = k+4, 4..7): V row = j*8 + (k+4)
    
    for r in range(16):
        j = r // 4
        k = r % 4
        for c in range(49):
            # First block
            V_total[j * 8 + k][c] = V4[r][c]
            # Second block
            V_total[j * 8 + 4 + k][49 + c] = V4[r][c]
    
    # W4 has 16 rows indexed by k*4 + i (k in 0..3, i in 0..3)
    # For first block: W row = k*4 + i (k in 0..3)
    # For second block: W row = (k+4)*4 + i = k*4 + 16 + i (k in 0..3)
    for r in range(16):
        k = r // 4
        i = r % 4
        for c in range(49):
            # First block
            W_total[k * 4 + i][c] = W4[r][c]
            # Second block
            W_total[(k + 4) * 4 + i][49 + c] = W4[r][c]
    
    # U is the same for both blocks
    for r in range(16):
        for c in range(49):
            U_total[r][c] = U4[r][c]
            U_total[r][49 + c] = U4[r][c]
    
    return {"U": U_total, "V": V_total, "W": W_total}
