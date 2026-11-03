import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define Strassen 2x2x2 decomposition
    # Input matrices X, Y (2x2)
    # Output Z = X*Y (2x2)
    # 7 products S_m
    # U_2, V_2, W_2 are 4x7 matrices
    # U_2[row, col] is coeff of X[row] in S_col
    # V_2[row, col] is coeff of Y[row] in S_col
    # W_2[row, col] is coeff of S_col in Z[row]
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # S0: (x00+x11)(y00+y11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    # S1: (x10+x11)y00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    # S2: x00(y01-y11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    # S3: x11(y10-y00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    # S4: (x00+x01)y11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    # S5: (x10-x00)(y00+y01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    # S6: (x01-x11)(y10+y11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    
    # Reconstruction of Z from S
    # Z00 = S0 - S2 + S4 - S5 + S6 ? 
    # Let's check standard Strassen formulas again.
    # c11 = M1 - M3 + M5 - M6 + M7
    # c12 = M4 - M2 + M5 - M7
    # c21 = M2 + M4 + M6 - M1
    # c22 = M1 - M4 + M3 - M5
    # Mapping M1..M7 to S0..S6
    # M1=S0, M2=S1, M3=S2, M4=S3, M5=S4, M6=S5, M7=S6
    
    # Z00 (c11) = S0 - S2 + S4 - S5 + S6
    W_2[0, 0] = 1; W_2[0, 2] = -1; W_2[0, 4] = 1; W_2[0, 5] = -1; W_2[0, 6] = 1
    # Z01 (c12) = S3 - S1 + S4 - S6
    W_2[1, 3] = 1; W_2[1, 1] = -1; W_2[1, 4] = 1; W_2[1, 6] = -1
    # Z10 (c21) = S1 + S3 + S5 - S0
    W_2[2, 1] = 1; W_2[2, 3] = 1; W_2[2, 5] = 1; W_2[2, 0] = -1
    # Z11 (c22) = S0 - S3 + S2 - S4
    W_2[3, 0] = 1; W_2[3, 3] = -1; W_2[3, 2] = 1; W_2[3, 4] = -1
    
    # Now define the 7 block products P_k for 4x4x4
    # P_k = X_k * Y_k
    # X_k is a linear combination of A blocks
    # Y_k is a linear combination of B blocks
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    
    # We need to map the "effective" 2x2 matrix X_k to the global 4x4 matrix A.
    # X_k is 2x2. Its elements are linear combinations of A's elements.
    # For example, if X_k = A00 + A11, then:
    # X_k[0,0] = A[0,0] + A[2,2]
    # X_k[0,1] = A[0,1] + A[2,3]
    # X_k[1,0] = A[1,0] + A[3,2]
    # X_k[1,1] = A[1,1] + A[3,3]
    
    # We will define for each P_k the coefficients for A and B.
    # A_coeff[k_p][block_idx] = coeff
    # B_coeff[k_p][block_idx] = coeff
    # block_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    A_specs = [
        [(1, 0), (1, 3)], # P0: A00 + A11
        [(1, 2), (1, 3)], # P1: A10 + A11
        [(1, 0)],         # P2: A00
        [(1, 3)],         # P3: A11
        [(1, 0), (1, 1)], # P4: A00 + A01
        [(1, 2), (-1, 0)],# P5: A10 - A00
        [(1, 1), (-1, 3)],# P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)], # P0: B00 + B11
        [(1, 0)],         # P1: B00
        [(1, 1), (-1, 3)],# P2: B01 - B11
        [(1, 2), (-1, 0)],# P3: B10 - B00
        [(1, 3)],         # P4: B11
        [(1, 0), (1, 1)], # P5: B00 + B01
        [(1, 2), (1, 3)], # P6: B10 + B11
    ]
    
    # C reconstruction
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # C_specs[k_p][block_idx] = coeff
    C_specs = [
        [(1, 0), (1, 3)], # P0 -> C00 + C11
        [(1, 2), (-1, 3)],# P1 -> C10 - C11
        [(1, 1), (1, 3)], # P2 -> C01 + C11
        [(1, 0), (1, 2)], # P3 -> C00 + C10
        [(-1, 0), (1, 1)],# P4 -> -C00 + C01
        [(1, 3)],         # P5 -> C11
        [(1, 0)],         # P6 -> C00
    ]
    
    def get_global_a_idx(block, r, c):
        # block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        bi = block // 2
        bj = block % 2
        return (bi * 2 + r) * 4 + (bj * 2 + c)
        
    def get_global_b_idx(block, r, c, b_offset):
        bi = block // 2
        bj = block % 2
        return (bi * 2 + r) * 8 + (bj * 2 + c + b_offset)
        
    def get_global_c_idx(block, r, c, c_offset):
        # C is 8x4. Index is k*n + i.
        # C_block (bi, bj) corresponds to rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1?
        # No, C is output.
        # C = A * B.
        # A is 4x4, B is 4x4 (subproblem).
        # C is 4x4.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # Block C00 is C[0:2, 0:2].
        # Block C01 is C[0:2, 2:4].
        # Block C10 is C[2:4, 0:2].
        # Block C11 is C[2:4, 2:4].
        # Global C index for subproblem half is (k_global * 4 + i_global).
        # k_global = 2*bi + r + c_offset
        # i_global = 2*bj + c
        bi = block // 2
        bj = block % 2
        return (2 * bi + r + c_offset) * 4 + (2 * bj + c)
        
    # Iterate over two halves
    for half in range(2):
        b_offset = half * 4
        c_offset = half * 4
        
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                # U (A)
                for (coeff, block) in A_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            # X_k[r, c] gets coeff * A_block[r, c]
                            # U_2[m_s] gives coeff of X_k[index] in S_m
                            # X_k index is r*2+c
                            u_coeff = U_2[r * 2 + c, m_s]
                            if u_coeff != 0:
                                a_idx = get_global_a_idx(block, r, c)
                                U[a_idx, col] += coeff * u_coeff
                                
                # V (B)
                for (coeff, block) in B_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            # Y_k[r, c] gets coeff * B_block[r, c]
                            # V_2[m_s] gives coeff of Y_k[index] in S_m
                            v_coeff = V_2[r * 2 + c, m_s]
                            if v_coeff != 0:
                                b_idx = get_global_b_idx(block, r, c, b_offset)
                                V[b_idx, col] += coeff * v_coeff
                                
                # W (C)
                for (coeff, block) in C_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            # Z_k[r, c] gets coeff * P_k[r, c]
                            # P_k[r, c] is sum_m W_2[r*2+c, m] * S_m
                            # So W_2[r*2+c, m_s] is the coeff of S_m in Z_k[r,c]
                            w_coeff = W_2[r * 2 + c, m_s]
                            if w_coeff != 0:
                                c_idx = get_global_c_idx(block, r, c, c_offset)
                                W[c_idx, col] += coeff * w_coeff
                                
    return {"U": U, "V": V, "W": W}
