import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Target rank: 85
    # 1. 4x4x4 (Strassen): 49
    # 2. 4x1x4 (Schoolbook): 16
    # 3. 4x4x1 (Schoolbook): 16
    # 4. 4x1x1 (Schoolbook): 4
    
    total_rank = 49 + 16 + 16 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal offset
        U[u_idx, offset] += u_val
        V[v_idx, offset] += v_val
        W[w_idx, offset] += w_val
        offset += 1

    # --- Part 1: 4x4x4 using Strassen ---
    # We need to generate the 49 terms for 4x4x4.
    # This is Strassen(4) = 7 * Strassen(2).
    
    # Strassen 2x2x2 terms (relative to block)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will iterate through the 7 top-level terms.
    # For each, we define the linear combination of A-blocks and B-blocks.
    # Then we decompose that 2x2x2 multiplication using Strassen 2x2x2.
    
    # Top level block indices: 0,1 for row, 0,1 for col.
    # A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
    # B is 4x4. Blocks B00, B01, B10, B11 are 2x2.
    # C is 4x4. Blocks C00, C01, C10, C11 are 2x2.
    
    # Strassen 2x2x2 decomposition function
    def get_strassen_2x2_terms():
        # Returns list of (u_coeffs, v_coeffs, w_coeffs) for 2x2x2
        # u: 4 entries (A00, A01, A10, A11)
        # v: 4 entries (B00, B01, B10, B11)
        # w: 4 entries (C00, C01, C10, C11)
        # Each is a list of 7 tuples? No, 7 terms.
        # Each term is a rank-1 tensor.
        terms = []
        # M1
        # A: 1,0,0,1 ; B: 1,0,0,1 ; C: 1,0,0,0 (C00 part of M1? No, M1 is 2x2)
        # Wait, Strassen 2x2x2 decomposes the tensor T.
        # The 7 terms are:
        # 1. (A00+A11) * (B00+B11) -> contributes to C00, C01, C10, C11?
        # No, the output of the 2x2x2 multiplication is a 2x2 matrix.
        # The decomposition is of the tensor T_{ijk}.
        # For 2x2x2, T has 8 non-zero entries.
        # The 7 rank-1 tensors sum to T.
        
        # Let's list the 7 rank-1 tensors for 2x2x2.
        # Indices 0,1 for i,j,k.
        # T[i,j,k] = 1 if i,j,k correspond to A_ij * B_jk -> C_ik.
        
        # Term 1: (A00+A11)(B00+B11)
        # This product is a 2x2 matrix X.
        # X_00 = A00 B00 + A00 B11 + A11 B00 + A11 B11 ? No.
        # Strassen computes M1 = (A00+A11)(B00+B11).
        # M1 is a scalar? No, A00 is a scalar in 2x2?
        # In 2x2x2, A is 2x2, B is 2x2, C is 2x2.
        # A00 is element (0,0).
        # So M1 is a scalar multiplication of two sums?
        # No, Strassen for 2x2x2 uses 7 scalar multiplications.
        # The "blocks" are scalars.
        
        # So for 2x2x2:
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # We need to express this as sum of rank-1 tensors U_r, V_r, W_r.
        # For each M_k, it is a product of a linear form in A and a linear form in B.
        # Let L_Ak = sum a_i A_i, L_Bk = sum b_j B_j.
        # M_k = L_Ak L_Bk.
        # The contribution to C is sum_k coeff_k M_k.
        # So C_c = sum_k coeff_k L_Ak L_Bk.
        # This can be written as sum_r U_r V_r W_r?
        # Yes, if we expand.
        
        # Actually, it's easier to just generate the U,V,W for the 7 terms directly.
        # For term k:
        # U vector has non-zeros where L_Ak has non-zeros.
        # V vector has non-zeros where L_Bk has non-zeros.
        # W vector has non-zeros where C receives contribution from M_k.
        
        # Let's define the 7 terms for 2x2x2.
        # Each term is (u, v, w) where u,v,w are dicts {index: coeff}.
        
        t1_u = {0:1, 3:1} # A00, A11 (indices 0,1,2,3 for 2x2)
        t1_v = {0:1, 3:1} # B00, B11
        t1_w = {0:1, 3:1} # C00, C11 ? No.
        # C00 gets +M1. C11 gets +M1.
        # So w1 = {0:1, 3:1}.
        
        t2_u = {2:1, 3:1} # A10, A11
        t2_v = {0:1} # B00
        t2_w = {2:1} # C10 gets +M2.
        
        t3_u = {0:1} # A00
        t3_v = {1:1, 3:-1} # B01, B11
        t3_w = {1:1} # C01 gets +M3.
        
        t4_u = {3:1} # A11
        t4_v = {2:1, 0:-1} # B10, B00
        t4_w = {0:1, 2:1} # C00 gets +M4, C10 gets +M4.
        
        t5_u = {0:1, 1:1} # A00, A01
        t5_v = {3:1} # B11
        t5_w = {0:-1, 1:1} # C00 gets -M5, C01 gets +M5.
        
        t6_u = {2:1, 0:-1} # A10, A00
        t6_v = {0:1, 1:1} # B00, B01
        t6_w = {3:1} # C11 gets +M6.
        
        t7_u = {1:1, 3:-1} # A01, A11
        t7_v = {2:1, 3:1} # B10, B11
        t7_w = {0:1, 3:1} # C00 gets +M7, C11 gets +M7.
        
        return [
            (t1_u, t1_v, t1_w),
            (t2_u, t2_v, t2_w),
            (t3_u, t3_v, t3_w),
            (t4_u, t4_v, t4_w),
            (t5_u, t5_v, t5_w),
            (t6_u, t6_v, t6_w),
            (t7_u, t7_v, t7_w)
        ]

    strassen_2x2 = get_strassen_2x2_terms()
    
    # Now for 4x4x4, we apply this to 2x2 blocks.
    # Top level Strassen terms M1..M7 involve sums of 2x2 blocks.
    # Each "multiplication" is a 2x2x2 multiplication.
    # So for each top-level term, we have 7 bottom-level terms.
    
    # Top level definitions (indices 0,1 for blocks):
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each top-level term Mk, we define the linear combination of A-blocks and B-blocks.
    # Then for each bottom-level term in strassen_2x2, we combine them.
    
    # We need to map block indices to global indices.
    # A block (bi, bj) where bi,bj in {0,1} corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # A element (i,j) in block (bi,bj) has global index (2*bi+i, 2*bj+j).
    
    def add_strassen_4x4():
        nonlocal offset
        top_terms = [
            # (u_blocks, v_blocks, w_blocks)
            # u_blocks: dict {block_idx: coeff}
            (
                {0:1, 3:1}, # A00, A11
                {0:1, 3:1}, # B00, B11
                {0:1, 3:1}  # C00, C11
            ),
            (
                {2:1, 3:1}, # A10, A11
                {0:1},      # B00
                {2:1}       # C10
            ),
            (
                {0:1},      # A00
                {1:1, 3:-1},# B01, B11
                {1:1}       # C01
            ),
            (
                {3:1},      # A11
                {2:1, 0:-1},# B10, B00
                {0:1, 2:1}  # C00, C10
            ),
            (
                {0:1, 1:1}, # A00, A01
                {3:1},      # B11
                {0:-1, 1:1} # C00, C01
            ),
            (
                {2:1, 0:-1},# A10, A00
                {0:1, 1:1}, # B00, B01
                {3:1}       # C11
            ),
            (
                {1:1, 3:-1},# A01, A11
                {2:1, 3:1}, # B10, B11
                {0:1, 3:1}  # C00, C11
            )
        ]
        
        for tu, tv, tw in top_terms:
            # For each bottom term
            for bu, bv, bw in strassen_2x2:
                # Combine
                # u_global = sum_{block, elem} tu[block] * bu[elem]
                # We add to U, V, W at offset.
                
                # Build U vector
                for b_idx, b_coeff in tu.items():
                    bi = b_idx // 2
                    bj = b_idx % 2
                    for e_idx, e_coeff in bu.items():
                        ei = e_idx // 2
                        ej = e_idx % 2
                        gi = 2*bi + ei
                        gj = 2*bj + ej
                        u_idx = gi * 4 + gj # n=4
                        U[u_idx, offset] += b_coeff * e_coeff
                        
                # Build V vector
                for b_idx, b_coeff in tv.items():
                    bi = b_idx // 2
                    bj = b_idx % 2
                    for e_idx, e_coeff in bv.items():
                        ei = e_idx // 2
                        ej = e_idx % 2
                        gi = 2*bi + ei
                        gj = 2*bj + ej
                        v_idx = gi * 4 + gj # m=4 for this part
                        V[v_idx, offset] += b_coeff * e_coeff
                        
                # Build W vector
                for b_idx, b_coeff in tw.items():
                    bi = b_idx // 2
                    bj = b_idx % 2
                    for e_idx, e_coeff in bw.items():
                        ei = e_idx // 2
                        ej = e_idx % 2
                        gi = 2*bi + ei
                        gj = 2*bj + ej
                        w_idx = gj * 4 + gi # p=4, n=4
                        W[w_idx, offset] += b_coeff * e_coeff
                        
                offset += 1

    add_strassen_4x4()
    
    # --- Part 2: 4x1x4 (Schoolbook) ---
    # A: col 4 (index 4)
    # B: row 4 (index 4), cols 0-3
    # C: cols 0-3
    for i in range(4):
        for k in range(4):
            # A[i, 4] * B[4, k] -> C[i, k]
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + k
            w_idx = k * 4 + i
            U[u_idx, offset] = 1
            V[v_idx, offset] = 1
            W[w_idx, offset] = 1
            offset += 1
            
    # --- Part 3: 4x4x1 (Schoolbook) ---
    # A: cols 0-3
    # B: col 4 (index 4)
    # C: col 4 (index 4)
    for i in range(4):
        for j in range(4):
            # A[i, j] * B[j, 4] -> C[i, 4]
            u_idx = i * 5 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i
            U[u_idx, offset] = 1
            V[v_idx, offset] = 1
            W[w_idx, offset] = 1
            offset += 1
            
    # --- Part 4: 4x1x1 (Schoolbook) ---
    # A: col 4
    # B: [4,4]
    # C: col 4
    for i in range(4):
        # A[i, 4] * B[4, 4] -> C[i, 4]
        u_idx = i * 5 + 4
        v_idx = 4 * 5 + 4
        w_idx = 4 * 4 + i
        U[u_idx, offset] = 1
        V[v_idx, offset] = 1
        W[w_idx, offset] = 1
        offset += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
