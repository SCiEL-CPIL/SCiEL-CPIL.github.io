"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    # Rank 28 using Strassen on 2x2 blocks
    rank = 28
    
    U = np.zeros((n * m, rank)) # 8 x 28
    V = np.zeros((m * p, rank)) # 16 x 28
    W = np.zeros((p * n, rank)) # 8 x 28
    
    # Helper to add a Strassen decomposition for X*Y=Z
    # X is a 2x2 block of A, Y is a 2x2 block of B, Z is a 2x2 block of C
    # start_col: starting column index in U, V, W
    def add_strassen(X_offsets, Y_offsets, Z_offsets, start_col):
        # X_offsets: list of 4 indices (i*m+j) for X[0,0], X[0,1], X[1,0], X[1,1]
        # Y_offsets: list of 4 indices (j*p+k) for Y[0,0], Y[0,1], Y[1,0], Y[1,1]
        # Z_offsets: list of 4 indices (k*n+i) for Z[0,0], Z[0,1], Z[1,0], Z[1,1]
        # Note: Z_offsets must correspond to the W indexing convention W[k*n+i]
        
        # Strassen coefficients for X,Y -> M1..M7
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y11)
        # M4 = X11(Y01-Y00)
        # M5 = (X00+X10)Y11
        # M6 = (X11-X00)(Y00+Y10)
        # M7 = (X01-X11)(Y11+Y01)
        
        # We define the contributions to U, V, W for each M
        # U corresponds to X, V to Y, W to Z (intermediate result storage? No, W is output C)
        # Wait, the decomposition is T = sum U_r V_r W_r.
        # So for each term M, we have:
        # U[:, col] = vector of coefficients for X
        # V[:, col] = vector of coefficients for Y
        # W[:, col] = vector of coefficients for Z (which is part of C)
        
        # However, Strassen computes M's which are then combined to get Z.
        # In the tensor decomposition, each "term" r is a rank-1 component.
        # So we create 7 terms for each 2x2 product.
        
        # Let's define the 7 terms.
        # Term 1: M1
        # U: X00=1, X11=1. Others 0.
        # V: Y00=1, Y11=1. Others 0.
        # W: This term contributes to Z. But in the tensor decomposition, W is the output vector.
        # The output C is a sum of these products.
        # So for term r, W[:, r] should be the vector that multiplies the scalar M_r to contribute to C.
        # But M_r is not an output, it's an intermediate.
        # The tensor decomposition is linear.
        # C = sum_r (U_r . A) * (V_r . B) * W_r ?? No.
        # The definition is: C = sum_r (U_r . A) * (V_r . B) * W_r is wrong.
        # The definition is: The tensor T is decomposed as sum_r u_r v_r w_r.
        # This means T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
        # Here T is the multiplication tensor.
        # T maps (A,B) to C.
        # So C[c] = sum_{a,b} A[a] B[b] T[a,b,c].
        # Substituting T: C[c] = sum_{a,b} A[a] B[b] sum_r U[a,r] V[b,r] W[c,r]
        # C[c] = sum_r W[c,r] (sum_a A[a] U[a,r]) (sum_b B[b] V[b,r])
        # Let s_r = sum_a A[a] U[a,r] and t_r = sum_b B[b] V[b,r].
        # Then C[c] = sum_r W[c,r] s_r t_r.
        
        # So for each Strassen step, we identify s_r and t_r as linear combinations of A and B.
        # And W[c,r] tells us how s_r t_r contributes to each C[c].
        
        # For a specific 2x2 product P = X Y:
        # X is a submatrix of A. Y is a submatrix of B.
        # P contributes to a submatrix Z of C.
        
        # Let's trace M1 = (X00+X11)(Y00+Y11).
        # s_1 = A[X00] + A[X11]
        # t_1 = B[Y00] + B[Y11]
        # Contribution to Z:
        # Z00 += M1
        # Z11 += M1
        # So W[Z00, r] += 1, W[Z11, r] += 1.
        
        # We will build the U, V, W matrices by iterating through the 4 blocks and 7 Strassen steps.
        
        steps = [
            # (X_coeffs, Y_coeffs, Z_coeffs)
            # X_coeffs: dict {idx: coeff}, Y_coeffs: dict {idx: coeff}, Z_coeffs: dict {idx: coeff}
            # Step 1: M1 = (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
            ({X_offsets[0]: 1, X_offsets[2]: 1}, {Y_offsets[0]: 1, Y_offsets[2]: 1}, {Z_offsets[0]: 1, Z_offsets[2]: 1}),
            # Step 2: M2 = (X10+X11)Y00 -> Z10 += 1
            ({X_offsets[1]: 1, X_offsets[2]: 1}, {Y_offsets[0]: 1}, {Z_offsets[1]: 1}),
            # Step 3: M3 = X00(Y10-Y11) -> Z01 += 1
            ({X_offsets[0]: 1}, {Y_offsets[1]: -1, Y_offsets[3]: 1}, {Z_offsets[3]: 1}), # Wait, Z01 is index 1?
            # Let's check Z indexing.
            # Z is 2x2. Z[0,0], Z[0,1], Z[1,0], Z[1,1].
            # Z_offsets[0] = Z[0,0], Z_offsets[1] = Z[0,1], Z_offsets[2] = Z[1,0], Z_offsets[3] = Z[1,1].
            # M3 contributes to Z01. So Z_offsets[1] += 1.
            # Correction to Step 3:
            # ({X_offsets[0]: 1}, {Y_offsets[1]: -1, Y_offsets[3]: 1}, {Z_offsets[1]: 1}),
            
            # Step 4: M4 = X11(Y01-Y00) -> Z00 += 1, Z10 += 1
            ({X_offsets[2]: 1}, {Y_offsets[1]: 1, Y_offsets[0]: -1}, {Z_offsets[0]: 1, Z_offsets[2]: 1}),
            
            # Step 5: M5 = (X00+X10)Y11 -> Z00 -= 1, Z01 += 1
            ({X_offsets[0]: 1, X_offsets[1]: 1}, {Y_offsets[3]: 1}, {Z_offsets[0]: -1, Z_offsets[1]: 1}),
            
            # Step 6: M6 = (X11-X00)(Y00+Y10) -> Z11 += 1
            ({X_offsets[2]: 1, X_offsets[0]: -1}, {Y_offsets[0]: 1, Y_offsets[2]: 1}, {Z_offsets[3]: 1}),
            
            # Step 7: M7 = (X01-X11)(Y11+Y01) -> Z00 += 1, Z11 -= 1
            ({X_offsets[3]: 1, X_offsets[2]: -1}, {Y_offsets[3]: 1, Y_offsets[1]: 1}, {Z_offsets[0]: 1, Z_offsets[3]: -1}),
        ]
        
        # Re-verify Strassen formulas:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y10-Y11)
        # M4 = X11(Y01-Y00)
        # M5 = (X00+X10)Y11
        # M6 = (X11-X00)(Y00+Y10)
        # M7 = (X01-X11)(Y11+Y01)
        
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # Let's re-map the steps to Z contributions correctly.
        
        # Step 1 (M1): Z00 += 1, Z11 += 1
        # Step 2 (M2): Z10 += 1, Z11 -= 1
        # Step 3 (M3): Z01 += 1, Z11 += 1
        # Step 4 (M4): Z00 += 1, Z10 += 1
        # Step 5 (M5): Z00 -= 1, Z01 += 1
        # Step 6 (M6): Z11 += 1
        # Step 7 (M7): Z00 += 1, Z11 -= 1
        
        # Let's re-write the steps list with correct Z coeffs.
        steps = [
            # 1
            ({X_offsets[0]: 1, X_offsets[2]: 1}, {Y_offsets[0]: 1, Y_offsets[2]: 1}, {Z_offsets[0]: 1, Z_offsets[3]: 1}),
            # 2
            ({X_offsets[1]: 1, X_offsets[2]: 1}, {Y_offsets[0]: 1}, {Z_offsets[2]: 1, Z_offsets[3]: -1}),
            # 3
            ({X_offsets[0]: 1}, {Y_offsets[2]: 1, Y_offsets[3]: -1}, {Z_offsets[1]: 1, Z_offsets[3]: 1}),
            # 4
            ({X_offsets[2]: 1}, {Y_offsets[1]: 1, Y_offsets[0]: -1}, {Z_offsets[0]: 1, Z_offsets[2]: 1}),
            # 5
            ({X_offsets[0]: 1, X_offsets[1]: 1}, {Y_offsets[3]: 1}, {Z_offsets[0]: -1, Z_offsets[1]: 1}),
            # 6
            ({X_offsets[2]: 1, X_offsets[0]: -1}, {Y_offsets[0]: 1, Y_offsets[2]: 1}, {Z_offsets[3]: 1}),
            # 7
            ({X_offsets[3]: 1, X_offsets[2]: -1}, {Y_offsets[3]: 1, Y_offsets[1]: 1}, {Z_offsets[0]: 1, Z_offsets[3]: -1}),
        ]
        
        for s in steps:
            x_coeff, y_coeff, z_coeff = s
            for idx, val in x_coeff.items():
                U[idx, start_col] = val
            for idx, val in y_coeff.items():
                V[idx, start_col] = val
            for idx, val in z_coeff.items():
                W[idx, start_col] = val
            start_col += 1
        return start_col
    
    # Define offsets for A (2x4)
    # A indices: i*m+j
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    
    # A_L (cols 0,1):
    # A_L[0,0]=0, A_L[0,1]=1, A_L[1,0]=4, A_L[1,1]=5
    A_L_off = [0, 1, 4, 5]
    # A_R (cols 2,3):
    # A_R[0,0]=2, A_R[0,1]=3, A_R[1,0]=6, A_R[1,1]=7
    A_R_off = [2, 3, 6, 7]
    
    # Define offsets for B (4x4)
    # B indices: j*p+k
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    # j=1, k=2 -> 6
    # j=1, k=3 -> 7
    # j=2, k=0 -> 8
    # j=2, k=1 -> 9
    # j=2, k=2 -> 10
    # j=2, k=3 -> 11
    # j=3, k=0 -> 12
    # j=3, k=1 -> 13
    # j=3, k=2 -> 14
    # j=3, k=3 -> 15
    
    # B_TL (rows 0,1 cols 0,1):
    # B_TL[0,0]=0, B_TL[0,1]=1, B_TL[1,0]=4, B_TL[1,1]=5
    B_TL_off = [0, 1, 4, 5]
    # B_TR (rows 0,1 cols 2,3):
    # B_TR[0,0]=2, B_TR[0,1]=3, B_TR[1,0]=6, B_TR[1,1]=7
    B_TR_off = [2, 3, 6, 7]
    # B_BL (rows 2,3 cols 0,1):
    # B_BL[0,0]=8, B_BL[0,1]=9, B_BL[1,0]=12, B_BL[1,1]=13
    B_BL_off = [8, 9, 12, 13]
    # B_BR (rows 2,3 cols 2,3):
    # B_BR[0,0]=10, B_BR[0,1]=11, B_BR[1,0]=14, B_BR[1,1]=15
    B_BR_off = [10, 11, 14, 15]
    
    # Define offsets for C (2x4)
    # C indices: k*n+i
    # k=0, i=0 -> 0 (C[0,0])
    # k=0, i=1 -> 1 (C[1,0])
    # k=1, i=0 -> 2 (C[0,1])
    # k=1, i=1 -> 3 (C[1,1])
    # k=2, i=0 -> 4 (C[0,2])
    # k=2, i=1 -> 5 (C[1,2])
    # k=3, i=0 -> 6 (C[0,3])
    # k=3, i=1 -> 7 (C[1,3])
    
    # C_LL (cols 0,1):
    # C_LL[0,0]=0, C_LL[0,1]=2, C_LL[1,0]=1, C_LL[1,1]=3
    # Note: Z_offsets expects [Z00, Z01, Z10, Z11]
    # Z00 is C[0,0] -> idx 0
    # Z01 is C[0,1] -> idx 2
    # Z10 is C[1,0] -> idx 1
    # Z11 is C[1,1] -> idx 3
    C_LL_off = [0, 2, 1, 3]
    
    # C_LR (cols 2,3):
    # C_LR[0,0]=4, C_LR[0,1]=6, C_LR[1,0]=5, C_LR[1,1]=7
    C_LR_off = [4, 6, 5, 7]
    
    # Execute the 4 blocks
    col = 0
    # P1 = A_L * B_TL -> C_LL
    col = add_strassen(A_L_off, B_TL_off, C_LL_off, col)
    # P2 = A_R * B_BL -> C_LL
    col = add_strassen(A_R_off, B_BL_off, C_LL_off, col)
    # P3 = A_L * B_TR -> C_LR
    col = add_strassen(A_L_off, B_TR_off, C_LR_off, col)
    # P4 = A_R * B_BR -> C_LR
    col = add_strassen(A_R_off, B_BR_off, C_LR_off, col)
    
    # Verify rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
