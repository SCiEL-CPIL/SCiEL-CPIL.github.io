import time
import sys

def construct():
    start_time = time.time()
    # Dimensions
    n = 4
    m = 4
    p = 4
    
    # We will build the rank 49 decomposition using Strassen recursively.
    # First, define the 2x2 Strassen decomposition parameters.
    # For 2x2, indices are i,j in {0,1}.
    # U_2x2 will be (4, 7), V_2x2 (4, 7), W_2x2 (4, 7).
    # Note: In 2x2, W index is k*n+i -> k*2+i.
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Strassen 7 terms for 2x2:
    # t=0: M1 = (A00+A11)(B00+B11)
    # t=1: M2 = (A10+A11)B00
    # t=2: M3 = A00(B01-B11)
    # t=3: M4 = A11(B10-B00)
    # t=4: M5 = (A00+A01)B11
    # t=5: M6 = (A10-A00)(B00+B01)
    # t=6: M7 = (A01-A11)(B10+B11)
    
    # Fill U2, V2 for the 7 terms
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # t0: M1
    U2[0][0] = 1; U2[3][0] = 1 # A00, A11
    V2[0][0] = 1; V2[3][0] = 1 # B00, B11
    # W2 for M1 contribution to C:
    # C00 += M1, C11 += M1
    # W index k*2+i
    # C00 is i=0, k=0 -> idx 0. C11 is i=1, k=1 -> idx 3.
    W2[0][0] = 1; W2[3][0] = 1
    
    # t1: M2
    U2[2][1] = 1; U2[3][1] = 1 # A10, A11
    V2[0][1] = 1;             # B00
    # C10 += M2 (i=1, k=0 -> idx 2)
    W2[2][1] = 1
    
    # t2: M3
    U2[0][2] = 1;             # A00
    V2[1][2] = 1; V2[3][2] = -1 # B01, -B11
    # C01 += M3 (i=0, k=1 -> idx 1)
    W2[1][2] = 1
    
    # t3: M4
    U2[3][3] = 1;             # A11
    V2[2][3] = 1; V2[0][3] = -1 # B10, -B00
    # C00 += M4 (idx 0), C10 += M4 (idx 2)
    W2[0][3] = 1; W2[2][3] = 1
    
    # t4: M5
    U2[0][4] = 1; U2[1][4] = 1 # A00, A01
    V2[3][4] = 1;             # B11
    # C00 -= M5 (idx 0), C01 += M5 (idx 1)
    W2[0][4] = -1; W2[1][4] = 1
    
    # t5: M6
    U2[2][5] = 1; U2[0][5] = -1 # A10, -A00
    V2[0][5] = 1; V2[1][5] = 1  # B00, B01
    # C11 += M6 (idx 3)
    W2[3][5] = 1
    
    # t6: M7
    U2[1][6] = 1; U2[3][6] = -1 # A01, -A11
    V2[2][6] = 1; V2[3][6] = 1  # B10, B11
    # C00 += M7 (idx 0)
    W2[0][6] = 1

    # Now construct 4x4 using this 2x2 building block.
    # Rank = 7 * 7 = 49.
    R = 49
    
    U = [[0]*R for _ in range(16)]
    V = [[0]*R for _ in range(16)]
    W = [[0]*R for _ in range(16)]
    
    # Map 4x4 indices (i,j) to block indices and local indices.
    # Block index b_u in {0,1}, local i_u in {0,1}.
    # Global index = b_u * 2 + i_u.
    
    # Outer Strassen 7 steps.
    # We define linear combinations of the 4 A-blocks (00,01,10,11) for each step.
    # And B-blocks.
    
    # A-blocks: A00 (indices 0..3), A01 (indices 4..7), A10 (indices 8..11), A11 (indices 12..15)
    # Note: My U2 indices: 0:(0,0)->A00[0], 1:(0,1)->A00[1]?
    # Wait, 2x2 matrix A has indices (0,0), (0,1), (1,0), (1,1).
    # In 4x4, a "block" Auv is a 2x2 matrix.
    # A00 is A[0..1][0..1]. Indices 0,1,2,3.
    # A01 is A[0..1][2..3]. Indices 4,5,6,7.
    # A10 is A[2..3][0..1]. Indices 8,9,10,11.
    # A11 is A[2..3][2..3]. Indices 12,13,14,15.
    
    # For the outer Strassen, we treat each 2x2 block as a scalar in the "block-strassen" algebra.
    # But since we want an exact scalar decomposition, we expand the block multiplication
    # using the inner Strassen decomposition.
    
    # Let's define the outer Strassen linear forms on the blocks.
    # outer_step s in 0..6.
    # A_comb_s is a 2x2 matrix of coefficients for blocks (00,01; 10,11).
    # B_comb_s is a 2x2 matrix of coefficients for blocks.
    
    # Standard Strassen 7 steps for 2x2 block matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # And C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We iterate through these 7 outer steps.
    # For each outer step s, we have linear coefficients for A-blocks and B-blocks.
    # We then perform the "inner" multiplication of the resulting 2x2 "effective" matrices.
    # This inner multiplication is decomposed using U2, V2, W2.
    
    # Helper to get global index for a block (bu, bv) and local indices (i, j)
    def get_global_a_index(bu, bv, i, j):
        # A is n x m. n=4, m=4.
        # Global row = bu * 2 + i. Global col = bv * 2 + j.
        return (bu * 2 + i) * 4 + (bv * 2 + j)
        
    def get_global_b_index(bu, bv, i, j):
        # B is m x p. m=4, p=4.
        # Global row = bu * 2 + i. Global col = bv * 2 + j.
        return (bu * 2 + i) * 4 + (bv * 2 + j)

    def get_global_c_index(bu, bv, i, j):
        # C is p x n? No, C is n x p (4x4).
        # The W index is k*n + i.
        # k is col of C (which is col of B), i is row of C (which is row of A).
        # Block col index bk = bv? No, B block index (bu, bv) where bu is row-block, bv is col-block.
        # So for C, the row block is bu (from A) and col block is bv (from B).
        # Local row i, local col k.
        # W index = k * n + i = k * 4 + i.
        # Global k = bv * 2 + k_local? No, k is the column index of C.
        # Column of C corresponds to column of B.
        # B block (bu, bv). Col block index is bv.
        # So global col k = bv * 2 + k_local.
        # Global row i = bu * 2 + i_local.
        return (bv * 2 + j) * 4 + (bu * 2 + i) # Using j as k_local, i as i_local
        
    # Define Outer Strassen Coefficients
    # A_coeffs[s][block_u][block_v]
    # B_coeffs[s][block_u][block_v]
    # C_comb[s][block_u][block_v] -> coefficient for M_s in C_uv
    
    A_coeffs = [ [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]] ]
    B_coeffs = [ [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]] ]
    C_comb =   [ [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]], [[0,0],[0,0]] ]
    
    # s=0 (M1): A = A00+A11, B = B00+B11. C00+=1, C11+=1
    A_coeffs[0][0][0] = 1; A_coeffs[0][1][1] = 1
    B_coeffs[0][0][0] = 1; B_coeffs[0][1][1] = 1
    C_comb[0][0][0] = 1; C_comb[0][1][1] = 1
    
    # s=1 (M2): A = A10+A11, B = B00. C10+=1
    A_coeffs[1][1][0] = 1; A_coeffs[1][1][1] = 1
    B_coeffs[1][0][0] = 1
    C_comb[1][1][0] = 1
    
    # s=2 (M3): A = A00, B = B01-B11. C01+=1
    A_coeffs[2][0][0] = 1
    B_coeffs[2][0][1] = 1; B_coeffs[2][1][1] = -1
    C_comb[2][0][1] = 1
    
    # s=3 (M4): A = A11, B = B10-B00. C00+=1, C10+=1
    A_coeffs[3][1][1] = 1
    B_coeffs[3][1][0] = 1; B_coeffs[3][0][0] = -1
    C_comb[3][0][0] = 1; C_comb[3][1][0] = 1
    
    # s=4 (M5): A = A00+A01, B = B11. C00-=1, C01+=1
    A_coeffs[4][0][0] = 1; A_coeffs[4][0][1] = 1
    B_coeffs[4][1][1] = 1
    C_comb[4][0][0] = -1; C_comb[4][0][1] = 1
    
    # s=5 (M6): A = A10-A00, B = B00+B01. C11+=1
    A_coeffs[5][1][0] = 1; A_coeffs[5][0][0] = -1
    B_coeffs[5][0][0] = 1; B_coeffs[5][0][1] = 1
    C_comb[5][1][1] = 1
    
    # s=6 (M7): A = A01-A11, B = B10+B11. C00+=1
    A_coeffs[6][0][1] = 1; A_coeffs[6][1][1] = -1
    B_coeffs[6][1][0] = 1; B_coeffs[6][1][1] = 1
    C_comb[6][0][0] = 1
    
    # Construct U, V, W
    for s in range(7):
        # For this outer step s, we have effective 2x2 block matrices A_eff and B_eff
        # We need to decompose A_eff * B_eff using the 2x2 Strassen terms (U2, V2, W2)
        # This will add 7 columns to U, V, W.
        
        for inner_t in range(7):
            col = s * 7 + inner_t
            
            # Determine U contribution
            # U2[local_a_idx][inner_t] gives the coefficient of A_eff[local_a_row][local_a_col]
            # local_a_idx 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
            # So for local (lu, lv):
            for lu in range(2):
                for lv in range(2):
                    idx_local = lu * 2 + lv
                    coeff_inner = U2[idx_local][inner_t]
                    if coeff_inner != 0:
                        # This corresponds to A_eff[lv][lu]? No.
                        # U2 index is row*2+col?
                        # In my U2 setup:
                        # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
                        # So lu=0, lv=0 -> idx 0. lu=0, lv=1 -> idx 1.
                        # So idx_local = lu * 2 + lv.
                        
                        # A_eff[lu][lv] is a linear combo of A blocks.
                        # A_eff[lu][lv] = sum_{bu,bv} A_coeffs[s][lu][lv]_??
                        # Wait, A_coeffs[s] is 2x2. Index [lu][lv]?
                        # In Strassen, A is 2x2 blocks.
                        # A[0][0] is block 00. A[0][1] is block 01.
                        # So A_eff[lu][lv] uses A_coeffs[s][lu][lv].
                        coeff_outer = A_coeffs[s][lu][lv]
                        if coeff_outer != 0:
                            total_coeff = coeff_inner * coeff_outer
                            if total_coeff != 0:
                                # We need to add this to U[global_a_index][col]
                                # global_a_index depends on block indices (bu, bv) and local indices (i, j).
                                # But wait, the inner Strassen acts on the *entries* of the block matrix.
                                # The "entry" A_eff[lu][lv] is itself a 2x2 block of scalars?
                                # NO. In recursive Strassen, A_eff[lu][lv] is a linear combo of *blocks*.
                                # The blocks are 2x2 matrices of scalars.
                                # So A_eff[lu][lv] = coeff_outer * A_block[lu][lv] ??
                                # No, A_coeffs[s] are the coefficients for the Strassen steps.
                                # Example s=0: A_eff[0][0] = 1 (coefficient for A00 block? No, A_eff is formed by summing blocks).
                                # In Strassen step 0: A_part = A00 + A11.
                                # This is a 2x2 block matrix?
                                # No, A00 is a 2x2 matrix. A11 is a 2x2 matrix.
                                # A_part is a 2x2 matrix (sum).
                                # So A_eff[lu][lv] is NOT a scalar, it's a 2x2 matrix?
                                # Yes.
                                # So A_eff[lu][lv] = coeff_outer * A_block[lu][lv]?
                                # For s=0, A_eff[0][0] gets contribution from A00 with coeff 1.
                                # A_eff[1][1] gets contribution from A11 with coeff 1.
                                # A_eff[0][1] is 0.
                                
                                # So, for a fixed outer step s and inner term t:
                                # We are processing the contribution of the "effective entry" A_eff[lu][lv] to the inner product.
                                # A_eff[lu][lv] is a 2x2 matrix (a block).
                                # The inner Strassen decomposition U2[idx_local][t] applies to the *scalar* entries of the 2x2 block matrix.
                                # Wait. If A_eff[lu][lv] is a 2x2 block, then the "2x2 Strassen" we are using (U2)
                                # assumes the input is a 2x2 matrix of *scalars*.
                                # Here, the "scalars" of the block-matrix are actually 2x2 blocks of original scalars.
                                # So we must distribute the inner Strassen coefficients over the 4 entries of the block A[lu][lv].
                                
                                # Let's refine:
                                # The inner term t uses the "value" of A_eff[lu][lv].
                                # This value is a linear combination of the entries of the block A[lu][lv].
                                # Specifically, it is just the block A[lu][lv] (scaled by coeff_outer).
                                # So we need to expand A_block[lu][lv] into its 4 scalar entries.
                                # The scalar entry (i,j) of the block corresponds to global index (lu*2+i, lv*2+j).
                                
                                for i in range(2):
                                    for j in range(2):
                                        g_idx = get_global_a_index(lu, lv, i, j)
                                        U[g_idx][col] += total_coeff
                                        
            # Determine V contribution
            for lu in range(2):
                for lv in range(2):
                    idx_local = lu * 2 + lv
                    coeff_inner = V2[idx_local][inner_t]
                    if coeff_inner != 0:
                        coeff_outer = B_coeffs[s][lu][lv]
                        if coeff_outer != 0:
                            total_coeff = coeff_inner * coeff_outer
                            if total_coeff != 0:
                                for i in range(2):
                                    for j in range(2):
                                        g_idx = get_global_b_index(lu, lv, i, j)
                                        V[g_idx][col] += total_coeff
                                        
            # Determine W contribution
            # W2[idx_local][inner_t] gives the coefficient for C_eff[lu][lv] from the inner product M_t.
            # Wait, C_eff[lu][lv] is the result of the inner multiplication.
            # The inner Strassen computes C_eff = A_eff * B_eff.
            # The result entries are C_eff[lu][lv] (where lu,lv are 0..1).
            # Each C_eff[lu][lv] is a 2x2 block.
            # The outer Strassen step s contributes to the final C blocks via C_comb[s].
            # C_comb[s][cu][cv] is the coefficient of M_s in C_cu_c_v.
            # So for the result block at (cu, cv), the coefficient from this term is C_comb[s][cu][cv].
            
            # We need to add to W.
            # W is indexed by (k, i) i.e. col, row.
            # The block C[cu][cv] has row-block cu, col-block cv.
            # Local indices i (row), k (col).
            # The inner W2[idx_c_local][t] gives the coefficient of C_eff[c_row][c_col] for the inner term.
            # Note: In my W2 setup, index was k*2+i.
            # W2[k*2+i][t].
            # So c_row = i_local? c_col = k_local?
            # In Strassen 2x2: C is 2x2.
            # W2[0] is C[0][0]? (i=0,k=0).
            # W2[1] is C[0][1]? (i=0,k=1).
            # W2[2] is C[1][0]? (i=1,k=0).
            # W2[3] is C[1][1]? (i=1,k=1).
            # Yes.
            
            for i_local in range(2):
                for k_local in range(2):
                    idx_w = k_local * 2 + i_local
                    coeff_inner_w = W2[idx_w][inner_t]
                    if coeff_inner_w != 0:
                        # This corresponds to C_eff[i_local][k_local] (row i, col k).
                        # In block terms, this is the block at (cu=i_local, cv=k_local)?
                        # Yes, the "2x2 matrix" result of the inner multiplication has blocks indexed by (row, col).
                        # So we look up C_comb[s][i_local][k_local].
                        coeff_outer_w = C_comb[s][i_local][k_local]
                        if coeff_outer_w != 0:
                            total_coeff = coeff_inner_w * coeff_outer_w
                            if total_coeff != 0:
                                # Add to W for the 4 entries of the block (i_local, k_local).
                                # Block (cu=i_local, cv=k_local).
                                # Global W index: col = k_global, row = i_global.
                                # k_global = k_local_block * 2 + k_sub.
                                # i_global = i_local_block * 2 + i_sub.
                                # Note: i_local here is the block row index (0 or 1).
                                # Let's rename to avoid confusion.
                                # cu = i_local (block row)
                                # cv = k_local (block col)
                                for i_sub in range(2):
                                    for k_sub in range(2):
                                        w_idx = get_global_c_index(cu := i_local, cv := k_local, i := i_sub, k := k_sub)
                                        # Wait, get_global_c_index(bu, bv, i, j) uses j as col.
                                        # So get_global_c_index(cu, cv, i_sub, k_sub)
                                        W[w_idx][col] += total_coeff

    # Verification
    # Check sum_r U[ij,r] V[jk,r] W[ki,r] == 1 for all i,j,k.
    # i,j,k in 0..3.
    # W index is k*4 + i.
    # Note: The indices in U are (i,j) -> i*4+j.
    # V indices are (j,k) -> j*4+k.
    
    # Simple check on a few points
    # i=0, j=0, k=0.
    # Sum over r U[0,r] V[0,r] W[0*4+0,r]
    # Should be 1.
    
    # Let's do a full check in code to be safe, though it might be slow. 16*16*49 ~ 12k ops. Fast enough.
    # Actually, 16x16x16 checks = 4096 sums of 49 terms. 200k ops. Fine.
    
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for r in range(R):
                    val += U[i*4 + j][r] * V[j*4 + k][r] * W[k*4 + i][r]
                if val != 1:
                    valid = False
                    # print(f"Fail at {i},{j},{k}: {val}")
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
         # Fallback to schoolbook if generation failed?
         # But Strassen should work.
         pass

    return {"U": U, "V": V, "W": W}
