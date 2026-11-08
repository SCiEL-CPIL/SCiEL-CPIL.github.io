"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 100
    
    # Rank 65: 49 (Strassen 4x4x4) + 16 (Schoolbook 4x4x1)
    rank = 65
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # --- Strassen Coefficients ---
    
    # Outer Strassen (4x4x4)
    # A blocks: 0:00, 1:01, 2:10, 3:11
    # B blocks: 0:00, 1:01, 2:10, 3:11
    # M1..M7 definitions: (A_side, B_side)
    # A_side is 4x4 matrix of coeffs for A0..A3
    # B_side is 4x4 matrix of coeffs for B0..B3
    
    outer_A = [
        [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], # M1: A0+A3
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]], # M2: A2+A3
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], # M3: A0
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], # M4: A3
        [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], # M5: A0+A1
        [[-1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], # M6: A2-A0
        [[0, 1, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # M7: A1-A3
    ]
    # Note: The above representation is slightly wrong. 
    # M1 = (A0+A3)(B0+B3). 
    # The "A_side" for M1 is the matrix X1 such that X1 = A0+A3.
    # X1 is a 2x2 block matrix.
    # X1[0,0] = A0, X1[0,1] = 0, X1[1,0] = 0, X1[1,1] = A3.
    # So for each block position (r,c) in the 2x2 block matrix, we have a vector of coeffs for A0..A3.
    
    # Let's define X_s as a 4x4 array where X_s[block_idx] is a vector of 4 coeffs.
    # block_idx 0:00, 1:01, 2:10, 3:11.
    
    def get_outer_coeffs(s):
        # Returns X_s (4 vectors of 4) and Y_s (4 vectors of 4)
        X = np.zeros((4, 4), dtype=int)
        Y = np.zeros((4, 4), dtype=int)
        if s == 0: # M1
            X[0] = [1, 0, 0, 0] # A0
            X[3] = [0, 0, 0, 1] # A3
            Y[0] = [1, 0, 0, 0] # B0
            Y[3] = [0, 0, 0, 1] # B3
        elif s == 1: # M2
            X[2] = [0, 0, 1, 0] # A2
            X[3] = [0, 0, 0, 1] # A3
            Y[0] = [1, 0, 0, 0] # B0
        elif s == 2: # M3
            X[0] = [1, 0, 0, 0] # A0
            Y[2] = [0, 0, 1, 0] # B2
            Y[3] = [0, 0, 0, -1] # -B3
        elif s == 3: # M4
            X[3] = [0, 0, 0, 1] # A3
            Y[1] = [0, 1, 0, 0] # B1
            Y[0] = [0, -1, 0, 0] # -B0
        elif s == 4: # M5
            X[0] = [1, 0, 0, 0] # A0
            X[1] = [0, 1, 0, 0] # A1
            Y[3] = [0, 0, 0, 1] # B3
        elif s == 5: # M6
            X[2] = [0, 0, 1, 0] # A2
            X[0] = [0, -1, 0, 0] # -A0
            Y[0] = [1, 0, 0, 0] # B0
            Y[1] = [0, 1, 0, 0] # B1
        elif s == 6: # M7
            X[1] = [0, 1, 0, 0] # A1
            X[3] = [0, 0, 0, -1] # -A3
            Y[2] = [0, 0, 1, 0] # B2
            Y[3] = [0, 0, 0, 1] # B3
        return X, Y

    # Outer C combination
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    outer_C = [
        [1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1], # C0: M1, M4, -M5, M7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], # C1: M3, M5
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # C2: M2, M4
        [1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # C3: M1, -M2, M3, M6
    ]
    # Note: outer_C[c_block][s] is coeff of M_s in C_c_block.
    # But we need to map M_s (which is 2x2 blocks) to C_c_block (which is 2x2 blocks).
    # The inner Strassen computes R = X Y. R has 4 blocks R0..R3.
    # M_s is a 2x2 matrix of 2x2 blocks.
    # So M_s corresponds to R0..R3 of the product X_s Y_s.
    # So C_c_block is a linear combination of the R blocks of M_s.
    # Specifically, C_c_block = sum_s coeff * R_s?
    # No, C is 4x4. C_c_block is a 2x2 block of C.
    # M_s is a 2x2 matrix of 2x2 blocks.
    # So C_c_block = sum_s c_{s, c} * M_s?
    # Yes, the outer Strassen formula is C = sum_s c_s M_s.
    # So C_c_block = sum_s c_{s, c} M_s.
    # And M_s is the result of X_s Y_s.
    # So the contribution of M_s to C_c_block is c_{s, c} * M_s.
    # M_s is a 2x2 matrix of 2x2 blocks.
    # So we need to add c_{s, c} * R_{s, r} to C_c_block?
    # No, M_s IS the 2x2 matrix of blocks.
    # So C_c_block = sum_s c_{s, c} M_s.
    # This means C_c_block[0,0] = sum_s c_{s, c} M_s[0,0].
    # M_s[0,0] is R0 of the product X_s Y_s.
    
    # Inner Strassen Coefficients (2x2x2)
    # X, Y are 2x2 matrices.
    # X0, X1, X2, X3 are blocks.
    # m1..m7
    # R0 = m1 + m4 - m5 + m7
    # R1 = m3 + m5
    # R2 = m2 + m4
    # R3 = m1 - m2 + m3 + m6
    
    # u_t[k] is coeff of X_k in m_t
    # v_t[k] is coeff of Y_k in m_t
    # w_t[r] is coeff of R_r in m_t
    
    inner_U = [
        [1, 0, 0, 1], # m1: X0+X3
        [0, 0, 1, 1], # m2: X2+X3
        [1, 0, 0, 0], # m3: X0
        [0, 0, 0, 1], # m4: X3
        [1, 1, 0, 0], # m5: X0+X1
        [-1, 0, 1, 0], # m6: X2-X0
        [0, 1, 0, -1]  # m7: X1-X3
    ]
    inner_V = [
        [1, 0, 0, 1], # m1: Y0+Y3
        [1, 0, 0, 0], # m2: Y0
        [0, 0, 1, -1], # m3: Y2-Y3
        [0, -1, 0, 0], # m4: Y1-Y0
        [0, 0, 0, 1], # m5: Y3
        [1, 1, 0, 0], # m6: Y0+Y1
        [0, 0, 1, 1]  # m7: Y2+Y3
    ]
    inner_W = [
        [1, 0, 0, 1], # R0: m1+m4-m5+m7
        [0, 0, 1, -1], # R1: m3+m5
        [0, 1, 0, 1], # R2: m2+m4
        [1, -1, 1, 0], # R3: m1-m2+m3+m6
        [-1, 1, 0, 0], # ... wait, inner_W is w_t[r]
        # Let's re-verify inner_W.
        # R0 = m1 + m4 - m5 + m7 -> w_1[0]=1, w_4[0]=1, w_5[0]=-1, w_7[0]=1
        # R1 = m3 + m5 -> w_3[1]=1, w_5[1]=1
        # R2 = m2 + m4 -> w_2[2]=1, w_4[2]=1
        # R3 = m1 - m2 + m3 + m6 -> w_1[3]=1, w_2[3]=-1, w_3[3]=1, w_6[3]=1
    ]
    # Correct inner_W (rows are m1..m7, cols are R0..R3)
    inner_W = np.array([
        [1, 0, 0, 1], # m1
        [0, 0, 1, -1], # m2
        [0, 1, 0, 1], # m3
        [1, 0, 1, 0], # m4
        [-1, 1, 0, 0], # m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 0]  # m7
    ], dtype=int)
    
    # Build the 49 terms
    col_idx = 0
    for s in range(7):
        X_s, Y_s = get_outer_coeffs(s)
        for t in range(7):
            # U for this term
            # U_global is 16 entries.
            # For each block k (0..3) in X_s, we have coeffs X_s[k] for A0..A3.
            # For each A_block a (0..3), we have 4 scalar entries.
            # u_t[k] is coeff of X_k in m_t.
            # So U_global[entry] = sum_{k} u_t[k] * X_s[k][a] * (scalar coeff of A_a in entry)
            
            # V_global is 16 entries.
            # V_global[entry] = sum_{k} v_t[k] * Y_s[k][b] * (scalar coeff of B_b in entry)
            
            # W_global is 16 entries.
            # W_global[entry] = sum_{r} w_t[r] * outer_C[c_block][s] * (scalar coeff of C_c_block in entry)
            # Note: outer_C[c_block][s] is the coeff of M_s in C_c_block.
            # M_s is a 2x2 matrix of blocks.
            # So C_c_block = sum_s outer_C[c_block][s] M_s.
            # M_s has blocks M_s[0]..M_s[3].
            # So C_c_block = sum_s outer_C[c_block][s] * (M_s[0] at pos 0, M_s[1] at pos 1, etc)?
            # No, M_s is a matrix.
            # C = sum_s c_s M_s.
            # So C[0,0] = sum_s c_s M_s[0,0].
            # C[0,1] = sum_s c_s M_s[0,1].
            # ...
            # So for a specific C entry (i, k), which corresponds to C_block cb and scalar pos sp:
            # C_entry = sum_s outer_C[cb][s] * M_s[cb_pos]?
            # No, C is 4x4. C_block cb is 2x2.
            # C = [C0 C1; C2 C3].
            # C0 is 2x2.
            # C = sum_s c_s M_s.
            # So C0 = sum_s c_s M_s[0,0]?
            # No, M_s is a 2x2 matrix of 2x2 blocks.
            # M_s = [M_s00 M_s01; M_s10 M_s11].
            # C = [C0 C1; C2 C3].
            # C0 = sum_s c_s M_s00.
            # C1 = sum_s c_s M_s01.
            # C2 = sum_s c_s M_s10.
            # C3 = sum_s c_s M_s11.
            
            # So for C_block cb (0..3), the coeff of M_s is outer_C[cb][s].
            # And M_s contributes its block M_s[cb] to C_cb.
            # M_s[cb] is the result of the inner Strassen term t for block cb.
            # The inner Strassen gives R_r for r=0..3.
            # M_s[cb] is R_cb of the product X_s Y_s.
            # So the coeff of R_cb in m_t is w_t[cb].
            # So W_global[entry] = outer_C[cb][s] * w_t[cb] * (scalar coeff of C_cb in entry).
            
            # Let's fill U, V, W
            for i in range(4):
                for j in range(4):
                    # U entry (i, j)
                    # i, j are global indices.
                    # Block a = i//2 * 2 + j//2?
                    # A is 4x4. Blocks are 2x2.
                    # A[i,j] is in block (i//2, j//2).
                    # Block index a = (i//2)*2 + (j//2).
                    a = (i // 2) * 2 + (j // 2)
                    # Scalar pos in block: (i%2, j%2).
                    # All 4 scalars in block a have the same coeff from X_s.
                    coeff = 0.0
                    for k in range(4):
                        coeff += inner_U[t][k] * X_s[k][a]
                    U[i * m + j, col_idx] = coeff
                    
            for j in range(4):
                for k in range(4):
                    # V entry (j, k)
                    # B is 4x4. Blocks 2x2.
                    # B[j,k] is in block (j//2, k//2).
                    b = (j // 2) * 2 + (k // 2)
                    coeff = 0.0
                    for k_idx in range(4):
                        coeff += inner_V[t][k_idx] * Y_s[k_idx][b]
                    V[j * p + k, col_idx] = coeff
            
            for k in range(4):
                for i in range(4):
                    # W entry (k, i)
                    # C is 4x4. Blocks 2x2.
                    # C[i,k] is in block (i//2, k//2).
                    # Block index cb = (i//2)*2 + (k//2).
                    cb = (i // 2) * 2 + (k // 2)
                    coeff = 0.0
                    coeff = outer_C[cb][s] * inner_W[t][cb]
                    W[k * n + i, col_idx] = coeff
                    
            col_idx += 1
            
    # Schoolbook for 5th column
    for i in range(4):
        for j in range(4):
            U[i * m + j, col_idx] = 1.0
            V[j * p + 4, col_idx] = 1.0
            W[4 * n + i, col_idx] = 1.0
            col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
