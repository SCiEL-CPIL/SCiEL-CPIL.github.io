import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W lists of lists
    # Start with empty lists
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a rank-7 Strassen decomposition for X (2x2) * Y (2x2) = Z (2x2)
    # u_indices: list of 4 indices for U rows corresponding to X_00, X_01, X_10, X_11
    # v_indices: list of 4 indices for V rows corresponding to Y_00, Y_01, Y_10, Y_11
    # w_indices: list of 4 indices for W rows corresponding to Z_00, Z_01, Z_10, Z_11
    
    def add_strassen(u_idx, v_idx, w_idx):
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        
        # M1 = X00 * (Y00 + Y11)
        # M2 = (X10 + X11) * Y00
        # M3 = (X00 + X01) * Y11
        # M4 = X11 * (Y10 + Y00)
        # M5 = (X00 + X11) * (Y00 + Y11)
        # M6 = (X01 - X11) * (Y10 + Y11)
        # M7 = (X10 - X00) * (Y00 + Y01)
        
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # We add 7 columns to U, V, W
        # Each column corresponds to one product Mk
        
        # M1
        U_rows.append({u_idx[0]: 1}) # X00
        V_rows.append({v_idx[0]: 1, v_idx[3]: 1}) # Y00 + Y11
        W_rows.append({w_idx[0]: 1, w_idx[1]: 0, w_idx[2]: 0, w_idx[3]: 1}) # Z00 += 1, Z11 += 1
        
        # M2
        U_rows.append({u_idx[2]: 1, u_idx[3]: 1}) # X10 + X11
        V_rows.append({v_idx[0]: 1}) # Y00
        W_rows.append({w_idx[2]: 1, w_idx[3]: -1}) # Z10 += 1, Z11 -= 1
        
        # M3
        U_rows.append({u_idx[0]: 1, u_idx[1]: 1}) # X00 + X01
        V_rows.append({v_idx[3]: 1}) # Y11
        W_rows.append({w_idx[1]: 1, w_idx[3]: 1}) # Z01 += 1, Z11 += 1
        
        # M4
        U_rows.append({u_idx[3]: 1}) # X11
        V_rows.append({v_idx[2]: 1, v_idx[0]: 1}) # Y10 + Y00
        W_rows.append({w_idx[0]: 1, w_idx[2]: 1}) # Z00 += 1, Z10 += 1
        
        # M5
        U_rows.append({u_idx[0]: 1, u_idx[3]: 1}) # X00 + X11
        V_rows.append({v_idx[0]: 1, v_idx[3]: 1}) # Y00 + Y11
        W_rows.append({w_idx[0]: -1, w_idx[1]: 1}) # Z00 -= 1, Z01 += 1
        
        # M6
        U_rows.append({u_idx[1]: 1, u_idx[3]: -1}) # X01 - X11
        V_rows.append({v_idx[2]: 1, v_idx[3]: 1}) # Y10 + Y11
        W_rows.append({w_idx[3]: 1}) # Z11 += 1
        
        # M7
        U_rows.append({u_idx[2]: 1, u_idx[0]: -1}) # X10 - X00
        V_rows.append({v_idx[0]: 1, v_idx[1]: 1}) # Y00 + Y01
        W_rows.append({w_idx[0]: 1}) # Z00 += 1

    # Define indices for the 4 blocks
    # A is 2x4. A0 is A[0:2, 0:2], A1 is A[0:2, 2:4]
    # U indices: i*m + j
    # A0: i in {0,1}, j in {0,1} -> 0, 1, 4, 5
    # A1: i in {0,1}, j in {2,3} -> 2, 3, 6, 7
    
    u_A0 = [0, 1, 4, 5] # X00, X01, X10, X11
    u_A1 = [2, 3, 6, 7]
    
    # B is 4x4.
    # B00: rows 0,1 cols 0,1. V indices: j*p + k
    # B00: j in {0,1}, k in {0,1} -> 0, 1, 4, 5
    # B01: j in {0,1}, k in {2,3} -> 2, 3, 6, 7
    # B10: j in {2,3}, k in {0,1} -> 8, 9, 12, 13
    # B11: j in {2,3}, k in {2,3} -> 10, 11, 14, 15
    
    v_B00 = [0, 1, 4, 5] # Y00, Y01, Y10, Y11
    v_B01 = [2, 3, 6, 7]
    v_B10 = [8, 9, 12, 13]
    v_B11 = [10, 11, 14, 15]
    
    # C is 2x4.
    # C0 is C[0:2, 0:2]. W indices: k*n + i
    # C0: k in {0,1}, i in {0,1} -> 0, 1, 2, 3
    # C1 is C[0:2, 2:4]. W indices: k*n + i
    # C1: k in {2,3}, i in {0,1} -> 4, 5, 6, 7
    
    w_C0 = [0, 1, 2, 3] # Z00, Z01, Z10, Z11
    w_C1 = [4, 5, 6, 7]
    
    # Block 1: A0 * B00 -> C0_left_part
    # C0 = A0 B00 + A1 B10
    # So A0 B00 contributes to C0
    add_strassen(u_A0, v_B00, w_C0)
    
    # Block 2: A1 * B10 -> C0_right_part
    add_strassen(u_A1, v_B10, w_C0)
    
    # Block 3: A0 * B01 -> C1_left_part
    # C1 = A0 B01 + A1 B11
    add_strassen(u_A0, v_B01, w_C1)
    
    # Block 4: A1 * B11 -> C1_right_part
    add_strassen(u_A1, v_B11, w_C1)
    
    # Convert dicts to dense matrices
    rank = len(U_rows)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for r, val in U_rows[c].items():
            U[r][c] = val
        for r, val in V_rows[c].items():
            V[r][c] = val
        for r, val in W_rows[c].items():
            W[r][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
