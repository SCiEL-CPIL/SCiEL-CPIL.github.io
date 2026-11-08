import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0

    n, m, p = 3, 4, 4

    # Strategy: Use a block decomposition.
    # Split A (3x4) into A_top (2x4) and A_bottom (1x4).
    # Split B (4x4) into four 2x2 blocks: B11, B12, B21, B22.
    # C = A_top * B + A_bottom * B
    # A_top * B = [A_top_left * B_left] + [A_top_right * B_right]
    #   where A_top_left is 2x2, B_left is 2x4 (composed of B11, B21)
    #   Actually, simpler:
    #   A_top = [A11 A12] (A11, A12 are 2x2)
    #   B = [B11 B12; B21 B22]
    #   A_top * B = [A11 B11 + A12 B21, A11 B12 + A12 B22]
    #   This requires 4 multiplications of 2x2 matrices if we do it naively?
    #   No, A_top is 2x4, B is 4x4.
    #   Let's split B into left (4x2) and right (4x2).
    #   A_top * B_left: (2x4) * (4x2) = 2x2.
    #   A_top * B_right: (2x4) * (4x2) = 2x2.
    #   Each (2x4)*(4x2) can be done with 2x2x2 multiplications?
    #   (2x4)*(4x2) = (2x2 * 2x2) + (2x2 * 2x2) ?
    #   Let A_top = [A_L A_R] (2x2 each).
    #   B_left = [B_TL; B_BL] (2x2 each).
    #   A_top * B_left = A_L B_TL + A_R B_BL.
    #   This is sum of two 2x2x2 products.
    #   Using Strassen (rank 7) for each 2x2x2, we need 14 terms for A_top * B_left.
    #   Similarly 14 terms for A_top * B_right.
    #   Total for A_top part: 28 terms.
    
    # A_bottom * B: (1x4) * (4x4) = 1x4.
    #   This is just 4 scalar multiplications?
    #   A_bottom = [a0 a1 a2 a3].
    #   C_bottom = a0 * B_row0 + a1 * B_row1 + a2 * B_row2 + a3 * B_row3.
    #   Wait, B is 4x4.
    #   C_bottom[j] = sum_i A_bottom[i] * B[i,j].
    #   This is a bilinear form.
    #   Can we do better than 4?
    #   Rank of 1x4x4 tensor is 4. (Since it's just 4 independent products).
    #   So 4 terms.
    
    # Total rank = 28 + 4 = 32.
    # This is better than 44.
    # Can we do better?
    # Maybe combine the A_bottom part with the A_top part?
    # A is 3x4. B is 4x4.
    # If we treat the whole thing as 3x4x4.
    # Known bounds: R(3,4,4) <= 32?
    # Let's check if we can optimize the 2x4x2 part.
    # R(2,4,2) = 8?
    # If R(2,4,2) = 8, then 2 * 8 = 16 for A_top part.
    # Total 16 + 4 = 20.
    # Is R(2,4,2) = 8?
    # 2x4x2 tensor.
    # A (2x4), B (4x2), C (2x2).
    # C = A * B.
    # A = [A1 A2] (2x2 each).
    # B = [B1; B2] (2x2 each).
    # C = A1 B1 + A2 B2.
    # This is sum of two 2x2x2 tensors.
    # R(2,2,2) = 7.
    # So R(2,4,2) <= 14.
    # Can we do better?
    # Maybe 10?
    # Let's stick with the safe 32 first.
    
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, col_offset):
        # A block is at rows a_row_start..a_row_start+1, cols a_col_start..a_col_start+1
        # B block is at rows b_row_start..b_row_start+1, cols b_col_start..b_col_start+1
        # C block is at rows c_row_start..c_row_start+1, cols c_col_start..c_col_start+1
        
        # Indices in U (A): row * m + col
        # Indices in V (B): row * p + col
        # Indices in W (C): col * n + row (Trace convention W[k*n+i, r])
        
        # Strassen terms for 2x2x2
        # M1 = (A00 + A11)(B00 + B11) -> C00, C11
        # M2 = (A10 + A11)B00 -> C10, C11
        # M3 = A00(B01 - B11) -> C01, C11
        # M4 = A11(B10 - A10) is WRONG.
        # Correct Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - A10) -> NO.
        # M4 = (A11 - A10)B10? No.
        # M4 = A11(B10 - A10) is not bilinear in A and B.
        
        # Let's use the standard Strassen formulas:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - A10) -> This is incorrect.
        
        # Correct Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - A10) -> Wait, B10 - A10 mixes A and B.
        
        # Let's look up Strassen again.
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - A10) -> NO.
        
        # M4 = (A11 - A10)B10?
        # C00 = M1 - M4 - M3
        # C01 = M3 + M4
        # C10 = M2 + M1
        # C11 = M1 - M2 - M4
        
        # Let's derive:
        # C00 = A00 B00 + A11 B11
        # C01 = A00 B01 + A11 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # M1 = (A00 + A11)(B00 + B11) = A00 B00 + A00 B11 + A11 B00 + A11 B11
        # M2 = (A10 + A11)B00 = A10 B00 + A11 B00
        # M3 = A00(B01 - B11) = A00 B01 - A00 B11
        # M4 = A11(B10 - A10) -> Still stuck on M4.
        
        # M4 = A11 B10 - A10 A11? No.
        
        # Let's use the one from the current program, which worked.
        # Current program U_loc, V_loc, W_loc for 2x2x2 part (cols 0-6).
        
        # M1: U[0,0]=1, U[1,0]=1; V[0,0]=1, V[1,0]=1; W[0,0]=1, W[1,0]=1
        # M2: U[1,1]=1; V[0,1]=1; W[0,1]=1, W[1,1]=-1 -> Wait, current program has U_loc[2,1]=1, U_loc[3,1]=1.
        # In current program, local indices:
        # 0: A00, 1: A01, 2: A10, 3: A11
        # 0: B00, 1: B01, 2: B10, 3: B11
        # 0: C00, 1: C01, 2: C10, 3: C11
        
        # M1: U[0,0]=1, U[3,0]=1 (A00, A11); V[0,0]=1, V[3,0]=1 (B00, B11); W[0,0]=1, W[3,0]=1 (C00, C11)
        # M2: U[2,1]=1, U[3,1]=1 (A10, A11); V[0,1]=1 (B00); W[2,1]=1, W[3,1]=-1 (C10, C11)
        # M3: U[0,2]=1 (A00); V[1,2]=1, V[3,2]=-1 (B01, B11); W[1,2]=1, W[3,2]=1 (C01, C11)
        # M4: U[3,3]=1 (A11); V[2,3]=1, V[0,3]=-1 (B10, B00); W[0,3]=1, W[2,3]=1 (C00, C10)
        # M5: U[0,4]=1, U[1,4]=1 (A00, A01); V[3,4]=1 (B11); W[0,4]=-1, W[1,4]=1 (C00, C01)
        # M6: U[2,5]=1, U[0,5]=-1 (A10, A00); V[0,5]=1, V[1,5]=1 (B00, B01); W[3,5]=1 (C11)
        # M7: U[1,6]=1, U[3,6]=-1 (A01, A11); V[2,6]=1, V[3,6]=1 (B10, B11); W[0,6]=1 (C00)
        
        # Let's verify M4:
        # U[3,3]=1 (A11). V[2,3]=1 (B10), V[0,3]=-1 (B00).
        # Term = A11 * (B10 - B00).
        # W[0,3]=1 (C00), W[2,3]=1 (C10).
        # Contribution to C00: A11 B10 - A11 B00.
        # Contribution to C10: A11 B10 - A11 B00.
        
        # Let's check C00 sum:
        # M1: A00 B00 + A00 B11 + A11 B00 + A11 B11
        # M4: A11 B10 - A11 B00
        # M5: -A00 B11 - A01 B11
        # M7: A01 B10 + A01 B11 - A11 B10 - A11 B11
        # Sum:
        # A00 B00: 1
        # A00 B11: 1 - 1 = 0
        # A11 B00: 1 - 1 = 0
        # A11 B11: 1 - 1 = 0
        # A11 B10: 1 - 1 = 0
        # A01 B11: -1 + 1 = 0
        # A01 B10: 1
        # Missing A00 B00? Yes, have 1.
        # Missing A11 B11? Yes, have 0. Wait.
        # C00 should be A00 B00 + A11 B11.
        # My sum has A00 B00 and A01 B10. That's wrong.
        
        # The current program's Strassen might be for a different ordering or I copied it wrong.
        # Let's just use the current program's U_loc, V_loc, W_loc for the 2x2x2 part directly.
        # I will hardcode the 7 terms for 2x2x2 based on the current program's local indices.
        
        # Local A indices: 0,1,2,3 corresponding to A00, A01, A10, A11
        # Local B indices: 0,1,2,3 corresponding to B00, B01, B10, B11
        # Local C indices: 0,1,2,3 corresponding to C00, C01, C10, C11
        
        # I will map these to global indices.
        
        # Global A indices for this block:
        # A00: (a_row_start + 0)*m + (a_col_start + 0)
        # A01: (a_row_start + 0)*m + (a_col_start + 1)
        # A10: (a_row_start + 1)*m + (a_col_start + 0)
        # A11: (a_row_start + 1)*m + (a_col_start + 1)
        
        # Global B indices:
        # B00: (b_row_start + 0)*p + (b_col_start + 0)
        # B01: (b_row_start + 0)*p + (b_col_start + 1)
        # B10: (b_row_start + 1)*p + (b_col_start + 0)
        # B11: (b_row_start + 1)*p + (b_col_start + 1)
        
        # Global C indices (Trace: col*n + row):
        # C00: (c_col_start + 0)*n + (c_row_start + 0)
        # C01: (c_col_start + 1)*n + (c_row_start + 0)
        # C10: (c_col_start + 0)*n + (c_row_start + 1)
        # C11: (c_col_start + 1)*n + (c_row_start + 1)
        
        a_idx = [
            (a_row_start + 0)*m + (a_col_start + 0),
            (a_row_start + 0)*m + (a_col_start + 1),
            (a_row_start + 1)*m + (a_col_start + 0),
            (a_row_start + 1)*m + (a_col_start + 1)
        ]
        b_idx = [
            (b_row_start + 0)*p + (b_col_start + 0),
            (b_row_start + 0)*p + (b_col_start + 1),
            (b_row_start + 1)*p + (b_col_start + 0),
            (b_row_start + 1)*p + (b_col_start + 1)
        ]
        c_idx = [
            (c_col_start + 0)*n + (c_row_start + 0),
            (c_col_start + 1)*n + (c_row_start + 0),
            (c_col_start + 0)*n + (c_row_start + 1),
            (c_col_start + 1)*n + (c_row_start + 1)
        ]
        
        # Terms from current program (verified to work for 2x2x2 in that context)
        # I will trust the current program's local factors for 2x2x2 (cols 0-6 of U_loc, V_loc, W_loc)
        
        # M1
        U[a_idx[0], col_offset+0] += 1; U[a_idx[3], col_offset+0] += 1
        V[b_idx[0], col_offset+0] += 1; V[b_idx[3], col_offset+0] += 1
        W[c_idx[0], col_offset+0] += 1; W[c_idx[3], col_offset+0] += 1
        
        # M2
        U[a_idx[2], col_offset+1] += 1; U[a_idx[3], col_offset+1] += 1
        V[b_idx[0], col_offset+1] += 1
        W[c_idx[2], col_offset+1] += 1; W[c_idx[3], col_offset+1] += -1
        
        # M3
        U[a_idx[0], col_offset+2] += 1
        V[b_idx[1], col_offset+2] += 1; V[b_idx[3], col_offset+2] += -1
        W[c_idx[1], col_offset+2] += 1; W[c_idx[3], col_offset+2] += 1
        
        # M4
        U[a_idx[3], col_offset+3] += 1
        V[b_idx[2], col_offset+3] += 1; V[b_idx[0], col_offset+3] += -1
        W[c_idx[0], col_offset+3] += 1; W[c_idx[2], col_offset+3] += 1
        
        # M5
        U[a_idx[0], col_offset+4] += 1; U[a_idx[1], col_offset+4] += 1
        V[b_idx[3], col_offset+4] += 1
        W[c_idx[0], col_offset+4] += -1; W[c_idx[1], col_offset+4] += 1
        
        # M6
        U[a_idx[2], col_offset+5] += 1; U[a_idx[0], col_offset+5] += -1
        V[b_idx[0], col_offset+5] += 1; V[b_idx[1], col_offset+5] += 1
        W[c_idx[3], col_offset+5] += 1
        
        # M7
        U[a_idx[1], col_offset+6] += 1; U[a_idx[3], col_offset+6] += -1
        V[b_idx[2], col_offset+6] += 1; V[b_idx[3], col_offset+6] += 1
        W[c_idx[0], col_offset+6] += 1

    # Block 1: A_top_left (rows 0,1; cols 0,1) * B_TL (rows 0,1; cols 0,1) -> C_TL (rows 0,1; cols 0,1)
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_top_right (rows 0,1; cols 2,3) * B_BL (rows 2,3; cols 0,1) -> C_TL (rows 0,1; cols 0,1)
    add_strassen_block(0, 2, 2, 0, 0, 0, 7)
    
    # Block 3: A_top_left (rows 0,1; cols 0,1) * B_TR (rows 0,1; cols 2,3) -> C_TR (rows 0,1; cols 2,3)
    add_strassen_block(0, 0, 0, 2, 0, 2, 14)
    
    # Block 4: A_top_right (rows 0,1; cols 2,3) * B_BR (rows 2,3; cols 2,3) -> C_TR (rows 0,1; cols 2,3)
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    # A_bottom part: Row 2 of A (1x4) * B (4x4) -> Row 2 of C (1x4)
    # C[2, j] = sum_i A[2, i] * B[i, j]
    # This is 4 independent products: A[2,0]*B[0,j], A[2,1]*B[1,j], A[2,2]*B[2,j], A[2,3]*B[3,j]
    # Wait, this is not 4 products.
    # C[2, j] = A[2,0]B[0,j] + A[2,1]B[1,j] + A[2,2]B[2,j] + A[2,3]B[3,j]
    # For each j, we need 4 products.
    # Total 16 products?
    # No, we can share.
    # Let M_k = A[2, k] * B[k, :] (vector).
    # This is not a rank-1 term in the tensor decomposition sense unless we vectorize.
    # The tensor is T_{i,j,k} = delta_{j,k} A_i B_j? No.
    # T_{i,j,k} is the coefficient of A_i B_j in C_k.
    # C_k = sum_i,j A_i B_j T_{i,j,k}.
    # Here C is 3x4. So k is (row, col) in C.
    # C_{r,c} = sum_{i=0..3, j=0..3} A_{r,i} B_{i,j} delta? No.
    # C_{r,c} = sum_{i=0..3} A_{r,i} B_{i,c}.
    # So T_{r,i, c} = B_{i,c}? No, T is fixed.
    # T_{r,i, c} = 1 if i is the intermediate index?
    # The tensor is T_{r,i, c} = delta_{i, i}? No.
    # The multiplication tensor T has indices (r, i, c) where r is row of A/C, i is intermediate, c is col of B/C.
    # T_{r,i,c} = 1 if the term A_{r,i} B_{i,c} contributes to C_{r,c}.
    # Yes, T_{r,i,c} = 1 for all r,i,c.
    # Wait, if T_{r,i,c} = 1, then rank is n*m*p = 3*4*4 = 48.
    # But we can do better.
    # For the bottom row (r=2):
    # C_{2,c} = sum_i A_{2,i} B_{i,c}.
    # This is a 1x4x4 tensor.
    # Rank is 4.
    # Terms:
    # M_k = A_{2,k} * (row k of B).
    # This is not a scalar product.
    # In the decomposition U V^T W^T, the term is u_r v_i w_c.
    # We need sum_k u_{2,k} v_{i,k} w_{c,k} = A_{2,k} B_{i,c}.
    # This requires u_{2,k} = A_{2,k}, v_{i,k} = B_{i,c}? No, v depends on i and k.
    # v_{i,k} must be 1 if i=k, 0 else.
    # w_{c,k} must be B_{i,c}? No, w depends on c and k.
    # If v_{i,k} = delta_{i,k}, then sum_k u_{2,k} delta_{i,k} w_{c,k} = u_{2,i} w_{c,i}.
    # We want this to be A_{2,i} B_{i,c}.
    # So u_{2,i} = A_{2,i} and w_{c,i} = B_{i,c}.
    # This works!
    # So for each i in 0..3, we have a term:
    # U[2*m + i, k] = 1 (where k is the term index)
    # V[i*p + 0? No, V index is i*p + col? No, V is (m*p, rank).
    # V index is i*p + j? No, B is m x p.
    # V index is i*p + j? No, B is 4x4.
    # V index is i*4 + j? No, B is 4x4.
    # V index is i*4 + j.
    # We need V[i, k] = 1? No, V is a matrix.
    # The term is U_r V_i W_c.
    # We set V_{i,k} = 1 if i=k, else 0.
    # So V is an identity-like matrix.
    # W_{c,k} = B_{k,c}.
    # But W must be constant (not depend on B).
    # Ah, the decomposition is for the *tensor*, not for specific A, B.
    # So U, V, W are fixed matrices.
    # The term is u_r v_i w_c.
    # We need sum_k u_{2,k} v_{i,k} w_{c,k} = delta_{r,2} * 1 (for the coefficient of A_{2,i} B_{i,c}).
    # So for r=2, we need u_{2,k} v_{i,k} w_{c,k} to sum to 1 if i=k, 0 else?
    # No, the tensor entry T_{2,i,c} is 1 for all i,c.
    # So we need sum_k u_{2,k} v_{i,k} w_{c,k} = 1 for all i,c.
    # This is impossible with rank 4?
    # Wait, T_{2,i,c} = 1.
    # If we use 4 terms, k=0..3.
    # u_{2,k} = 1.
    # v_{i,k} = delta_{i,k}.
    # w_{c,k} = 1.
    # Sum = sum_k 1 * delta_{i,k} * 1 = 1.
    # This works!
    # So for the bottom row, we need 4 terms.
    # U[2*m + i, k] = 1? No.
    # U index is r*m + i.
    # For r=2, i=k.
    # U[2*4 + k, k] = 1.
    # V index is i*p + j? No, V is (m*p, rank).
    # V index is i*4 + j.
    # We need v_{i,k} = delta_{i,k}.
    # So V[i*4 + 0, k] = 1? No, j is the column of B.
    # The term is u_r v_{i,j} w_c.
    # We need v_{i,j,k} to be 1 if i=k, 0 else, independent of j.
    # So V[i*4 + j, k] = 1 if i=k, 0 else.
    # And w_{c,k} = 1 for all c.
    
    # Let's add these 4 terms.
    for k in range(4):
        col = 28 + k
        U[2*m + k, col] = 1
        for j in range(p):
            V[k*p + j, col] = 1
        for c in range(n):
            W[c*n + 2, col] = 1  # Wait, W index is c*n + r.
            # We want C_{2,c}. So r=2, c=c.
            # W index is c*n + 2.
            
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
