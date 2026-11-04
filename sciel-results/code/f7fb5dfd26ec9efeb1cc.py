import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Rank 98 Strassen-based decomposition
    # 2 halves of B (4x4 each)
    # Each half uses 7 Strassen products
    # Each product uses 7 terms from 2x2x2 Strassen
    rank = 98
    
    # Initialize U, V, W
    # U: (n*m, rank) = (16, 98)
    # V: (m*p, rank) = (32, 98)
    # W: (p*n, rank) = (32, 98)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition details
    # We decompose a 2x2 matrix product X*Y = Z
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]
    
    # Indices in the flattened 4-vector:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # U_2, V_2, W_2 are 4x7 matrices
    # U_2 maps X to the 7 intermediate scalars
    # V_2 maps Y to the 7 intermediate scalars
    # W_2 maps the 7 intermediate scalars to Z
    
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # Standard Strassen formulas for 2x2x2
    # S0 = (x00+x11)(y00+y11)
    # S1 = (x10+x11)y00
    # S2 = x00(y01-y11)
    # S3 = x11(y10-y00)
    # S4 = (x00+x01)y11
    # S5 = (x10-x00)(y00+y01)
    # S6 = (x01-x11)(y10+y11)
    
    # z00 = S0 + S3 - S4 + S6
    # z01 = S2 + S4
    # z10 = S1 + S3
    # z11 = S0 - S1 + S2 + S5
    
    # m=0 (S0)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # m=1 (S1)
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # m=2 (S2)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # m=3 (S3)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # m=4 (S4)
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # m=5 (S5)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # m=6 (S6)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Block Strassen coefficients for 4x4x4
    # We split A (4x4) into 2x2 blocks of 2x2.
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B is 4x8. We split it into two 4x4 matrices: B_L (cols 0-3) and B_R (cols 4-7).
    # For each half, we split B_half into 2x2 blocks of 2x2.
    
    # A_specs: How to form A_eff for each of the 7 Strassen products P_k
    # P_k = A_eff_k * B_eff_k
    # A_eff_k is a 2x2 matrix formed from linear combinations of A's blocks.
    # A_eff_k[r, c] = sum coeff * A_block[r, c]
    
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    # B_specs: How to form B_eff for each P_k
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]
    
    # C_specs: How P_k contributes to C blocks
    # C = A * B
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(1, 2), (-1, 3)],  # P1: C10 - C11
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]
    
    # Iterate over the two halves of B (L and R)
    for half in range(2):
        if time.time() > deadline:
            break
        # half 0: cols 0-3 of B
        # half 1: cols 4-7 of B
        
        for k_p in range(7):
            if time.time() > deadline:
                break
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s
                
                # --- U (Input A) ---
                # U has shape (16, 98). Index: (i*m + j) where A is (i,j)
                # A is 4x4.
                # A_eff is 2x2.
                # A_eff[r, c] corresponds to A_block[r, c] in the 2x2 block structure of A.
                # Global A index for block (bi, bj) and local (r, c) is:
                # i_global = bi*2 + r
                # j_global = bj*2 + c
                # flat_idx = i_global * 4 + j_global
                
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            i_global = bi * 2 + r
                            j_global = bj * 2 + c
                            global_a_idx = i_global * 4 + j_global
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                # --- V (Input B) ---
                # V has shape (32, 98). Index: (j*p + k) where B is (j,k)
                # B is 4x8.
                # B_eff is 2x2.
                # B_eff[r, c] corresponds to B_block[r, c] in the 2x2 block structure of B_half.
                # Global B index for block (bi, bj) and local (r, c) is:
                # j_global = bi*2 + r
                # k_local = bj*2 + c
                # k_global = k_local + half*4
                # flat_idx = j_global * 8 + k_global
                
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            j_global = bi * 2 + r
                            k_local = bj * 2 + c
                            k_global = k_local + half * 4
                            global_b_idx = j_global * 8 + k_global
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                # --- W (Output C) ---
                # W has shape (32, 98). Index: (k*n + i) where C is (i,k)
                # C is 4x8.
                # C_eff is 2x2 (result of A_eff * B_eff).
                # C_eff[r, c] corresponds to C_block[r, c] in the 2x2 block structure of C_half.
                # Global C index for block (bi, bj) and local (r, c) is:
                # i_global = bi*2 + r
                # k_local = bj*2 + c
                # k_global = k_local + half*4
                # flat_idx = k_global * 4 + i_global
                
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            i_global = bi * 2 + r
                            k_local = bj * 2 + c
                            k_global = k_local + half * 4
                            global_w_idx = k_global * 4 + i_global
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]
                
                if time.time() > deadline:
                    break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
