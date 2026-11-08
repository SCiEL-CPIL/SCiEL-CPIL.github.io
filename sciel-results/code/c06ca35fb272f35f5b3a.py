import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n, m, p = 3, 4, 4

    # Strategy: Decompose 3x4x4 into 1x4x4 and 2x4x4 subproblems.
    # Rank(1x4x4) = 16 (Direct multiplication: C_j = sum_i A_i B_ij, tensor rank is m*p=16)
    # Rank(2x4x4) = 28 (Two 2x2x4 blocks, each rank 14 via two 2x2x2 Strassen blocks)
    # Total Rank = 16 + 28 = 44.
    
    # This is a known good bound. We construct the factors explicitly.
    
    U = np.zeros((n * m, 44), dtype=float)  # 12 x 44
    V = np.zeros((m * p, 44), dtype=float)  # 16 x 44
    W = np.zeros((p * n, 44), dtype=float)  # 12 x 44
    
    # Part 1: Compute C[0, :] = A[0, :] * B[0:4, 0:4]
    # This is a 1x4x4 multiplication.
    # C_k = sum_{j=0}^3 A_j B_{jk}
    # We need 16 terms, one for each (j, k) pair.
    # Term t corresponds to j = t // 4, k = t % 4.
    # U[A_j, t] = 1
    # V[B_{jk}, t] = 1
    # W[C_k, t] = 1
    
    for j in range(4):
        for k in range(4):
            t = j * 4 + k
            # A index: row 0, col j -> 0*4 + j = j
            U[j, t] = 1.0
            # B index: row j, col k -> j*4 + k
            V[j * 4 + k, t] = 1.0
            # C index: col k, row 0 -> k*3 + 0 = 3k
            W[3 * k, t] = 1.0
    
    # Part 2: Compute C[1:3, :] = A[1:3, :] * B[0:4, 0:4]
    # This is a 2x4x4 multiplication.
    # Let A' = A[1:3, :] (2x4), C' = C[1:3, :] (2x4).
    # We split B into two 2x4 blocks: B_top = B[0:2, :], B_bot = B[2:4, :].
    # And A' into two 2x2 blocks: A'_left = A'[0:2, 0:2], A'_right = A'[0:2, 2:4].
    # Wait, A' is 2x4. Let's split columns of A' and rows of B.
    # Actually, standard block decomposition:
    # A' = [A00, A01] where A00, A01 are 2x2.
    # B = [B00, B01; B10, B11] where blocks are 2x2.
    # C' = [A00 B00 + A01 B10, A00 B01 + A01 B11]
    # We need to compute 4 products of 2x2 matrices.
    # Each 2x2x2 multiplication has rank 7 (Strassen).
    # Total rank 4 * 7 = 28.
    
    # Let's define the Strassen terms for a generic 2x2x2 multiplication.
    # Input matrices X (2x2), Y (2x2). Output Z (2x2).
    # Z_00, Z_01, Z_10, Z_11.
    # X indices: 0,1,2,3 (row-major)
    # Y indices: 0,1,2,3 (row-major)
    # Z indices: 0,1,2,3 (row-major)
    
    # Strassen factors for 2x2x2:
    # We need U_s (4x7), V_s (4x7), W_s (4x7)
    # such that Z = U_s (X . V_s) W_s^T ? No, the tensor decomposition is:
    # Z_{ij} = sum_k U_{ik} V_{jk} W_{ij,k} ? No.
    # The tensor T_{i, j, k} for matmul is defined by C_{ij} = sum_{k,l} A_{ik} B_{kl} T_{ik, kl, ij}.
    # Here indices are:
    # A index: i, k
    # B index: k, l
    # C index: i, j
    # The term is A_{ik} B_{kl}. It contributes to C_{ij} if k=l? No, k is the sum index.
    # Wait, the standard tensor decomposition for M(2,2,2) is:
    # C_{ij} = sum_{t=1}^7 U_{i,t} V_{j,t} W_{t} ? No.
    # The factors are U (4x7), V (4x7), W (4x7).
    # The bilinear form is sum_t U[a_idx, t] * V[b_idx, t] * W[c_idx, t].
    # Where a_idx is linear index of A, b_idx of B, c_idx of C.
    
    # Let's use the known Strassen factors.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Indices:
    # A: 0:a00, 1:a01, 2:a10, 3:a11
    # B: 0:b00, 1:b01, 2:b10, 3:b11
    # C: 0:c00, 1:c01, 2:c10, 3:c11
    
    # Strassen terms:
    # t0: (a00+a11)(b00+b11) -> c00, c11
    # t1: (a10+a11)b00 -> c10, c11
    # t2: a00(b10-b00) -> c01, c11
    # t3: a11(b01-b11) -> c00, c10
    # t4: (a00+a01)b11 -> c00, c01
    # t5: (a10-a00)(b00+b10) -> c11
    # t6: (a01-a11)(b01+b11) -> c00
    
    # Let's build U_s, V_s, W_s for these terms.
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # t0
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # t1
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    
    # t2
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # t3
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # t4
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # t5
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # t6
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # Now map these to the global 2x4x4 problem.
    # A' is 2x4. Indices in A':
    # 0: A'[0,0] (global A[1,0])
    # 1: A'[0,1] (global A[1,1])
    # 2: A'[0,2] (global A[1,2])
    # 3: A'[0,3] (global A[1,3])
    # 4: A'[1,0] (global A[2,0])
    # 5: A'[1,1] (global A[2,1])
    # 6: A'[1,2] (global A[2,2])
    # 7: A'[1,3] (global A[2,3])
    # Global A index: row * 4 + col.
    # A'[r, c] is at global A[(1+r)*4 + c].
    
    # B is 4x4. Indices in B:
    # 0: B[0,0]
    # 1: B[0,1]
    # 2: B[0,2]
    # 3: B[0,3]
    # 4: B[1,0]
    # 5: B[1,1]
    # 6: B[1,2]
    # 7: B[1,3]
    # 8: B[2,0]
    # 9: B[2,1]
    # 10: B[2,2]
    # 11: B[2,3]
    # 12: B[3,0]
    # 13: B[3,1]
    # 14: B[3,2]
    # 15: B[3,3]
    # Global B index: row * 4 + col.
    
    # C' is 2x4. Indices in C':
    # 0: C'[0,0] (global C[0,1] -> index 1*3+0 = 3? No. C is 4x3.
    # Global C index: col * 3 + row.
    # C'[r, c] is at global C[c, 1+r].
    # Global index: c * 3 + (1+r).
    
    # We have 4 blocks of 2x2x2.
    # Block 0: A00 * B00 -> C00_part
    # A00 is A'[0:2, 0:2]. B00 is B[0:2, 0:2]. C00_part is C'[0:2, 0:2].
    # Block 1: A00 * B01 -> C01_part
    # A00 is A'[0:2, 0:2]. B01 is B[0:2, 2:4]. C01_part is C'[0:2, 2:4].
    # Block 2: A01 * B10 -> C10_part
    # A01 is A'[0:2, 2:4]. B10 is B[2:4, 0:2]. C10_part is C'[0:2, 0:2].
    # Block 3: A01 * B11 -> C11_part
    # A01 is A'[0:2, 2:4]. B11 is B[2:4, 2:4]. C11_part is C'[0:2, 2:4].
    
    # Let's define the mappings for each block.
    # For each block, we need to map the local 2x2 indices to global indices.
    
    blocks = [
        # (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
        # a_row_off is 0 for A', so global row is 1 + a_row.
        (0, 0, 0, 0, 0, 0), # Block 0
        (0, 0, 0, 2, 0, 2), # Block 1
        (0, 2, 2, 0, 0, 0), # Block 2
        (0, 2, 2, 2, 0, 2)  # Block 3
    ]
    
    for b_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
        t_off = 16 + b_idx * 7  # Start after the 1x4x4 part
        
        for t in range(7):
            global_t = t_off + t
            
            # Map U_s (A indices) to global U
            for local_a in range(4):
                if U_s[local_a, t] != 0:
                    # local_a is 0..3.
                    # local_a = r * 2 + c
                    r = local_a // 2
                    c = local_a % 2
                    global_a_idx = (1 + ar + r) * 4 + (ac + c)
                    U[global_a_idx, global_t] += U_s[local_a, t]
            
            # Map V_s (B indices) to global V
            for local_b in range(4):
                if V_s[local_b, t] != 0:
                    r = local_b // 2
                    c = local_b % 2
                    global_b_idx = (br + r) * 4 + (bc + c)
                    V[global_b_idx, global_t] += V_s[local_b, t]
            
            # Map W_s (C indices) to global W
            for local_c in range(4):
                if W_s[local_c, t] != 0:
                    r = local_c // 2
                    c = local_c % 2
                    # Global C index: (cc + c) * 3 + (1 + cr + r)
                    global_c_idx = (cc + c) * 3 + (1 + cr + r)
                    W[global_c_idx, global_t] += W_s[local_c, t]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
