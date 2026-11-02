import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2 parameters
    # P_s(X) for X = [[x00, x01], [x10, x11]]
    # Indices 0,1,2,3 for x00, x01, x10, x11
    P_2x2 = [
        [1, 0, 0, 1],  # P1
        [0, 0, 0, 1],  # P2
        [1, 1, 0, 0],  # P3
        [1, 0, 0, 0],  # P4
        [0, 0, 1, 1],  # P5
        [1, 0, -1, 0], # P6
        [0, 1, 0, -1]  # P7
    ]
    
    # Q_s(Y) for Y = [[y00, y01], [y10, y11]]
    Q_2x2 = [
        [1, 0, 0, 1],  # Q1
        [1, 1, 0, 0],  # Q2
        [0, 0, 0, 1],  # Q3
        [0, 0, 1, -1], # Q4
        [1, 0, 0, 0],  # Q5
        [0, 1, 0, 1],  # Q6
        [1, 0, 1, 0]   # Q7
    ]
    
    # Z_s matrices 2x2
    # Z1 = [[1,0],[0,1]]
    # Z2 = [[0,0],[1,1]]
    # Z3 = [[0,1],[0,-1]]
    # Z4 = [[1,0],[0,0]]
    # Z5 = [[-1,0],[1,0]]
    # Z6 = [[0,1],[0,1]]
    # Z7 = [[1,0],[0,0]]
    Z_2x2 = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[0, 1], [0, -1]],
        [[1, 0], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    # Top level Strassen 4x4
    # L_k(A) linear forms on 4x4 blocks
    # A blocks: A11 (0:2, 0:2), A12 (0:2, 2:4), A21 (2:4, 0:2), A22 (2:4, 2:4)
    # We represent L_k(A) as a 4x4 matrix of coefficients? No, it's a linear combination of blocks.
    # Each block is 2x2.
    # L_k(A) is a 2x2 matrix of linear forms on A's entries.
    # Actually, L_k(A) is a 4x4 matrix where each entry is a linear form on A.
    # But since L_k(A) is formed by adding/subtracting 2x2 blocks, the entries are just 1, -1, or 0.
    
    # Let's define the 4x4 coefficient matrices for L_k(A)
    # L1 = A11 + A22
    L_4x4 = [
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], # L1
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], # L2
        [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], # L3
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], # L4
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], # L5
        [[1, 1, -1, -1], [1, 1, -1, -1], [0, 0, 0, 0], [0, 0, 0, 0]], # L6
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, -1, -1], [0, 0, -1, -1]]  # L7
    ]
    
    # R_k(B) linear forms on 4x4 blocks
    # R1 = B11 + B22
    R_4x4 = [
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], # R1
        [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], # R2
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], # R3
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, -1, -1], [1, 1, -1, -1]], # R4
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], # R5
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], # R6
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]  # R7
    ]
    
    # Coefficients of M_k in C blocks
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M6
    # C21 = M2 + M5
    # C22 = M1 - M3 + M2 + M6
    # Indices: 0=C11, 1=C12, 2=C21, 3=C22
    C_coeffs = [
        [1, 0, 0, 1],  # k=1
        [0, 0, 1, 1],  # k=2
        [0, 1, 0, -1], # k=3
        [1, 0, 0, 0],  # k=4
        [0, 0, 1, 0],  # k=5 -- Wait, C11 has -M5.
        [0, 1, 0, 1],  # k=6
        [1, 0, 0, 0]   # k=7
    ]
    # Correct C_coeffs:
    # k=1: C11=1, C22=1
    # k=2: C21=1, C22=1
    # k=3: C12=1, C22=-1
    # k=4: C11=1
    # k=5: C11=-1, C21=1
    # k=6: C12=1, C22=1
    # k=7: C11=1
    
    C_coeffs = [
        [1, 0, 0, 1], # k=1
        [0, 0, 1, 1], # k=2
        [0, 1, 0, -1],# k=3
        [1, 0, 0, 0], # k=4
        [-1, 0, 1, 0],# k=5
        [0, 1, 0, 1], # k=6
        [1, 0, 0, 0]  # k=7
    ]
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for half in range(2):
        for k in range(7):
            for s in range(7):
                # Compute U vector for this term
                # P_s(L_k(A))
                # L_k(A) is 4x4 coeffs.
                # P_s is linear form on 2x2 blocks of L_k(A)?
                # No, L_k(A) is a 4x4 matrix.
                # P_s is defined on 2x2 matrices.
                # L_k(A) is composed of 2x2 blocks.
                # Let L_k(A) blocks be L_k(11), L_k(12), L_k(21), L_k(22).
                # Each L_k(xy) is a 2x2 matrix of coeffs from A.
                # P_s(L_k(A)) = P_s( [L_k(11) L_k(12); L_k(21) L_k(22)] )
                # This results in a 4x4 matrix of coeffs for A.
                
                # Let's compute the 4x4 coefficient matrix for A
                # U_coeff is 4x4
                U_coeff = [[0]*4 for _ in range(4)]
                for i in range(4):
                    for j in range(4):
                        # Determine which block of L_k(A) this (i,j) belongs to
                        bi = i // 2
                        bj = j // 2
                        # Get the 2x2 block L_k(bi, bj)
                        # L_k is defined by adding blocks of A.
                        # We stored L_4x4[k] as 4x4 coeffs.
                        # So L_k(i,j) is L_4x4[k][i][j].
                        # But P_s operates on the 2x2 structure of L_k.
                        # The 2x2 blocks of L_k are:
                        # Block 00: L_4x4[k][0:2, 0:2]
                        # Block 01: L_4x4[k][0:2, 2:4]
                        # Block 10: L_4x4[k][2:4, 0:2]
                        # Block 11: L_4x4[k][2:4, 2:4]
                        
                        # We need to apply P_s to these 4 blocks?
                        # No, P_s is a linear form on the 4 entries of a 2x2 matrix.
                        # L_k(A) is a 4x4 matrix.
                        # The "entries" of the 2x2 matrix of blocks are the blocks themselves.
                        # So P_s(L_k) = P_s( [B00, B01; B10, B11] )
                        # = c11*B00 + c12*B01 + c21*B10 + c22*B11
                        # where c11, etc are from P_s vector.
                        
                        # So for each (i,j) in 4x4, it belongs to a specific block of the 2x2 block structure.
                        # And within that block, it has a specific position.
                        
                        # Let's iterate over the 4 blocks of the 2x2 structure
                        for bb_i in range(2):
                            for bb_j in range(2):
                                # P_s coefficient for this block
                                # P_s is indexed by s (0..6)
                                # P_s vector is [p00, p01, p10, p11]
                                # Block index in P_s: bb_i * 2 + bb_j
                                p_val = P_2x2[s][bb_i * 2 + bb_j]
                                if p_val == 0:
                                    continue
                                # The block L_k(bb_i, bb_j) is a 2x2 matrix of coeffs from A
                                # For each entry in this 2x2 block:
                                for di in range(2):
                                    for dj in range(2):
                                        i_idx = bb_i * 2 + di
                                        j_idx = bb_j * 2 + dj
                                        # The coeff in L_k at (i_idx, j_idx)
                                        l_val = L_4x4[k][i_idx][j_idx]
                                        # Total coeff for A[i_idx, j_idx]
                                        U_coeff[i_idx][j_idx] += p_val * l_val
                        
                # Similarly for V
                V_coeff = [[0]*4 for _ in range(4)]
                for i in range(4):
                    for j in range(4):
                        for bb_i in range(2):
                            for bb_j in range(2):
                                q_val = Q_2x2[s][bb_i * 2 + bb_j]
                                if q_val == 0:
                                    continue
                                for di in range(2):
                                    for dj in range(2):
                                        i_idx = bb_i * 2 + di
                                        j_idx = bb_j * 2 + dj
                                        r_val = R_4x4[k][i_idx][j_idx]
                                        V_coeff[i_idx][j_idx] += q_val * r_val
                                        
                # Fill U, V, W
                for i in range(4):
                    for j in range(4):
                        val = U_coeff[i][j]
                        if val != 0:
                            U[i * m + j][col] = val
                        val = V_coeff[i][j]
                        if val != 0:
                            # B is 4x8.
                            # half 0: cols 0-3
                            # half 1: cols 4-7
                            # V index: j * p + k
                            # Here i is row of B (0..3), j is col of B (0..3) within the half
                            # Actual col in B: j + half*4
                            v_idx = i * p + (j + half * 4)
                            V[v_idx][col] = val
                            
                # Fill W
                # Z_s is 2x2
                # C_coeffs[k] is [c11, c12, c21, c22]
                # C block 00 (C11) corresponds to rows 0-1, cols 0-1 (for half 0)
                # C block 01 (C12) corresponds to rows 0-1, cols 2-3 (for half 0)
                # C block 10 (C21) corresponds to rows 2-3, cols 0-1 (for half 0)
                # C block 11 (C22) corresponds to rows 2-3, cols 2-3 (for half 0)
                
                for cb_i in range(2):
                    for cb_j in range(2):
                        # Coeff from C_coeffs
                        # cb_i=0, cb_j=0 -> C11
                        # cb_i=0, cb_j=1 -> C12
                        # cb_i=1, cb_j=0 -> C21
                        # cb_i=1, cb_j=1 -> C22
                        c_val = C_coeffs[k][cb_i * 2 + cb_j]
                        if c_val == 0:
                            continue
                        
                        # Z_s entry for this block
                        # Z_s is 2x2.
                        # The entry at (cb_i, cb_j) of the 2x2 block structure?
                        # No, Z_s is the matrix added to M_k.
                        # M_k is 2x2 blocks.
                        # Z_s is a 2x2 matrix.
                        # So Z_s[cb_i][cb_j] is the scalar multiplier for the block?
                        # No, Z_s is the matrix itself.
                        # M_k = sum P_s Q_s Z_s.
                        # Z_s is 2x2.
                        # So the contribution to M_k[cb_i, cb_j] is P_s Q_s Z_s[cb_i, cb_j].
                        z_val = Z_2x2[s][cb_i][cb_j]
                        if z_val == 0:
                            continue
                        
                        total_w = c_val * z_val
                        
                        # Now distribute this to C entries
                        # Block (cb_i, cb_j) of C (4x4) corresponds to:
                        # Rows: cb_i*2 to cb_i*2+1
                        # Cols: cb_j*2 to cb_j*2+1
                        for di in range(2):
                            for dj in range(2):
                                i_idx = cb_i * 2 + di
                                j_idx = cb_j * 2 + dj
                                # C is 4x8.
                                # Col in C: j_idx + half*4
                                c_col = j_idx + half * 4
                                # W index: k * n + i
                                # Here k is col of C, i is row of C
                                w_idx = c_col * n + i_idx
                                W[w_idx][col] = total_w
                                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
