import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
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
    
    # Outer Strassen terms (k=0..6)
    # X_k is linear comb of A_blocks
    # Y_k is linear comb of B_blocks
    
    # A_blocks: A00, A01, A10, A11 (each 2x2)
    # Indices for A: 
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # We define the linear forms for X_k and Y_k in terms of the 4 blocks.
    # Then we decompose each block product into 7 scalar products.
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeff):
        # u_indices: list of (flat_index, coeff)
        # v_indices: list of (flat_index, coeff)
        # w_indices: list of (flat_index, coeff)
        pass

    # This manual construction is complex. 
    # Alternative: Use the known Laderman decomposition directly if possible.
    # Or, implement the recursive logic carefully.
    
    # Let's try to generate the 49 terms programmatically.
    
    # 1. Define the 7 outer combinations
    # x_comb[k][block] = coeff
    # y_comb[k][block] = coeff
    
    # Blocks: 0:A00, 1:A01, 2:A10, 3:A11
    x_comb = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [1, 0, 1, 0], # M6: A10 - A00 -> [-1, 0, 1, 0] ? No, A10 - A00. 
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    # Correction for M6: (A10 - A00). So A00 coeff -1, A10 coeff 1.
    x_comb[5] = [-1, 0, 1, 0]
    
    y_comb = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00 -> [-1, 0, 1, 0]
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    y_comb[3] = [-1, 0, 1, 0]
    
    # Outer reconstruction coefficients
    # C00: M1, M4, -M5, M7
    # C01: M3, M5
    # C10: M2, M4
    # C11: M1, -M2, M3, M6
    
    # Block index to global index mapping
    # Block 0 (A00): rows 0-1, cols 0-1
    # Block 1 (A01): rows 0-1, cols 2-3
    # Block 2 (A10): rows 2-3, cols 0-1
    # Block 3 (A11): rows 2-3, cols 2-3
    
    def block_to_indices(b):
        r = b // 2
        c = b % 2
        start_r = r * 2
        start_c = c * 2
        return [start_r * 4 + start_c, start_r * 4 + start_c + 1, (start_r + 1) * 4 + start_c, (start_r + 1) * 4 + start_c + 1]

    # Inner Strassen on 2x2 matrices
    # Input X, Y (2x2)
    # 7 products
    # We need the linear forms for the 7 products in terms of the 4 entries of X and Y.
    
    # Inner Strassen forms:
    # S1 = X00 + X11, T1 = Y00 + Y11
    # S2 = X10 + X11, T2 = Y00
    # S3 = X00, T3 = Y01 - Y11
    # S4 = X11, T4 = Y10 - Y00
    # S5 = X00 + X01, T5 = Y11
    # S6 = X10 - X00, T6 = Y00 + Y01
    # S7 = X01 - X11, T7 = Y10 + Y11
    
    # Reconstruction of Z (2x2) from M1..M7
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We will iterate through outer k, and inner l.
    # For each (k, l), we have a scalar product S_{k,l}(X_k) * T_{k,l}(Y_k).
    # S_{k,l}(X_k) is a linear combination of the entries of X_k.
    # X_k is a 2x2 matrix of blocks? No, X_k is a linear combination of blocks.
    # Wait, X_k is a 2x2 matrix?
    # In the outer step, X_k is a "scalar" (block combination).
    # But to apply inner Strassen, we treat X_k as a 2x2 matrix.
    # This implies X_k must be a 2x2 matrix.
    # But X_k = A00 + A11 is a 2x2 matrix.
    # So the "entries" of X_k for the inner Strassen are the scalar entries of the 2x2 matrix X_k.
    
    # So, for a fixed k:
    # X_k is a 2x2 matrix.
    # Let X_k entries be x00, x01, x10, x11.
    # These are linear combinations of A's entries.
    # x00 = sum(c_i * A_i)
    
    # Inner Strassen term l uses linear forms of x00..x11.
    # So S_{k,l} is a linear combination of x00..x11.
    # Which is a linear combination of A's entries.
    
    # We need to map the block contributions to the scalar entries.
    # X_k = sum_b x_comb[k][b] * A_b
    # A_b is a 2x2 matrix.
    # So X_k is a 2x2 matrix.
    # X_k[r,c] = sum_b x_comb[k][b] * A_b[r,c]
    
    # Inner Strassen forms (l=0..6) on X_k:
    # s_forms[l][entry] where entry is 0..3 (x00, x01, x10, x11)
    s_forms = [
        [1, 0, 0, 1], # 1: x00 + x11
        [0, 0, 1, 1], # 2: x10 + x11
        [1, 0, 0, 0], # 3: x00
        [0, 0, 0, 1], # 4: x11
        [1, 1, 0, 0], # 5: x00 + x01
        [-1, 0, 1, 0], # 6: x10 - x00
        [0, 1, 0, -1]  # 7: x01 - x11
    ]
    
    t_forms = [
        [1, 0, 0, 1], # 1: y00 + y11
        [1, 0, 0, 0], # 2: y00
        [0, 1, 0, -1], # 3: y01 - y11
        [-1, 0, 1, 0], # 4: y10 - y00
        [0, 0, 0, 1], # 5: y11
        [1, 1, 0, 0], # 6: y00 + y01
        [0, 0, 1, 1]  # 7: y10 + y11
    ]
    
    # Inner reconstruction (Z entries from M1..M7)
    # z_recon[entry][l]
    z_recon = [
        [1, 0, 0, 1, -1, 0, 1], # Z00: M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # Z01: M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # Z10: M2 + M4
        [1, -1, 1, 0, 0, 1, 0]   # Z11: M1 - M2 + M3 + M6
    ]
    
    # Outer reconstruction (C blocks from M1..M7)
    # c_recon[block][k]
    c_recon = [
        [1, 0, 0, 1, -1, 0, 1], # C00: M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # C01: M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10: M2 + M4
        [1, -1, 1, 0, 0, 1, 0]   # C11: M1 - M2 + M3 + M6
    ]
    
    term_idx = 0
    for k in range(7):
        # Build X_k and Y_k linear forms on A and B
        # X_k is 2x2 matrix.
        # X_k[r,c] = sum_b x_comb[k][b] * A_b[r,c]
        
        # For each inner term l
        for l in range(7):
            # S is linear form on x00..x11
            # T is linear form on y00..y11
            
            # U vector (coeffs for A)
            u_coeffs = [0] * 16
            for entry in range(4):
                coeff_s = s_forms[l][entry]
                if coeff_s == 0: continue
                # entry corresponds to X_k[entry//2, entry%2]
                r = entry // 2
                c = entry % 2
                # X_k[r,c] = sum_b x_comb[k][b] * A_b[r,c]
                for b in range(4):
                    coeff_x = x_comb[k][b]
                    if coeff_x == 0: continue
                    total_coeff = coeff_s * coeff_x
                    # A_b[r,c] indices
                    a_indices = block_to_indices(b)
                    a_idx = a_indices[entry] # Wait, A_b is 2x2. entry is 0..3.
                    # A_b[0,0] is a_indices[0], A_b[0,1] is a_indices[1], etc.
                    # entry 0 -> r=0,c=0 -> A_b[0,0]
                    # entry 1 -> r=0,c=1 -> A_b[0,1]
                    # entry 2 -> r=1,c=0 -> A_b[1,0]
                    # entry 3 -> r=1,c=1 -> A_b[1,1]
                    # So a_indices[entry] is correct.
                    u_coeffs[a_indices[entry]] += total_coeff
            
            v_coeffs = [0] * 16
            for entry in range(4):
                coeff_t = t_forms[l][entry]
                if coeff_t == 0: continue
                r = entry // 2
                c = entry % 2
                for b in range(4):
                    coeff_y = y_comb[k][b]
                    if coeff_y == 0: continue
                    total_coeff = coeff_t * coeff_y
                    b_indices = block_to_indices(b)
                    v_coeffs[b_indices[entry]] += total_coeff
            
            # W vector (coeffs for C)
            w_coeffs = [0] * 16
            # Contribution to C comes from Z (which is M_k)
            # M_k contributes to C blocks via c_recon
            for cb in range(4): # C block
                coeff_c = c_recon[cb][k]
                if coeff_c == 0: continue
                # Z (M_k) is 2x2.
                # Z[entry] contributes to C_cb[entry]
                for entry in range(4):
                    coeff_z = z_recon[entry][l]
                    if coeff_z == 0: continue
                    total_coeff = coeff_c * coeff_z
                    c_indices = block_to_indices(cb)
                    w_coeffs[c_indices[entry]] += total_coeff
            
            # Add to U, V, W
            for i in range(16):
                if u_coeffs[i] != 0:
                    U[i][term_idx] = u_coeffs[i]
            for j in range(16):
                if v_coeffs[j] != 0:
                    V[j][term_idx] = v_coeffs[j]
            for ki in range(16):
                if w_coeffs[ki] != 0:
                    W[ki][term_idx] = w_coeffs[ki]
            
            term_idx += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res)
