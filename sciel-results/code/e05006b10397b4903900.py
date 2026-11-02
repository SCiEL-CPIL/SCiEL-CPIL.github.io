"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Rank 49 for 4x4x4 part (Strassen)
    # Rank 16 for 4x4x1 part (Schoolbook)
    rank = 49 + 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for 4x4x4 ---
    # We need the 2x2x2 Strassen decomposition coefficients
    # U2, V2, W2 of shape (4, 7)
    
    # Define the 7 terms for 2x2x2
    # z1 = (a00+a11)(b00+b11)
    # z2 = (a10+a11)b00
    # z3 = a00(b01-b11)
    # z4 = a11(b10-b00)
    # z5 = (a00+a01)b11
    # z6 = (a10-a00)(b00+b01)
    # z7 = (a01-a11)(b10+b11)
    
    # Output:
    # c00 = z1 + z4 - z5 + z7
    # c01 = z3 + z5
    # c10 = z2 + z4
    # c11 = z1 - z3 + z6 + z7
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Helper to set coefficients
    # Indices for 2x2: i,j in {0,1}. Row = i*2+j
    def set2(mat, i, j, val, col):
        mat[i*2+j][col] = val

    # z1
    set2(U2, 0, 0, 1, 0); set2(U2, 1, 1, 1, 0)
    set2(V2, 0, 0, 1, 0); set2(V2, 1, 1, 1, 0)
    set2(W2, 0, 0, 1, 0); set2(W2, 1, 1, 1, 0) # c00, c11
    
    # z2
    set2(U2, 1, 0, 1, 1); set2(U2, 1, 1, 1, 1)
    set2(V2, 0, 0, 1, 1)
    set2(W2, 0, 1, 1, 1) # c10
    
    # z3
    set2(U2, 0, 0, 1, 2)
    set2(V2, 0, 1, 1, 2); set2(V2, 1, 1, -1, 2)
    set2(W2, 1, 0, 1, 2); set2(W2, 1, 1, -1, 2) # c01, c11
    
    # z4
    set2(U2, 1, 1, 1, 3)
    set2(V2, 1, 0, 1, 3); set2(V2, 0, 0, -1, 3)
    set2(W2, 0, 0, 1, 3); set2(W2, 0, 1, 1, 3) # c00, c10
    
    # z5
    set2(U2, 0, 0, 1, 4); set2(U2, 0, 1, 1, 4)
    set2(V2, 1, 1, 1, 4)
    set2(W2, 0, 0, -1, 4); set2(W2, 1, 0, 1, 4) # c00, c01
    
    # z6
    set2(U2, 1, 0, 1, 5); set2(U2, 0, 0, -1, 5)
    set2(V2, 0, 0, 1, 5); set2(V2, 0, 1, 1, 5)
    set2(W2, 1, 1, 1, 5) # c11
    
    # z7
    set2(U2, 0, 1, 1, 6); set2(U2, 1, 1, -1, 6)
    set2(V2, 1, 0, 1, 6); set2(V2, 1, 1, 1, 6)
    set2(W2, 0, 0, 1, 6); set2(W2, 1, 1, 1, 6) # c00, c11

    # High-level Strassen coefficients for 4x4x4
    # Blocks: 00, 01, 10, 11
    # L_r coefficients for A blocks
    # R_r coefficients for B blocks
    # E_r coefficients for C blocks (where M_r goes)
    
    # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    # M2 = (A10+A11)B00       -> C10(+1)
    # M3 = A00(B01-B11)       -> C01(+1), C11(-1)
    # M4 = A11(B10-B00)       -> C00(+1), C10(+1)
    # M5 = (A00+A01)B11       -> C00(-1), C01(+1)
    # M6 = (A10-A00)(B00+B01) -> C11(+1)
    # M7 = (A01-A11)(B10+B11) -> C00(+1), C11(+1)
    
    L_coeffs = [
        {0: 1, 3: 1}, # M1: A00, A11
        {2: 1, 3: 1}, # M2: A10, A11
        {0: 1},       # M3: A00
        {3: 1},       # M4: A11
        {0: 1, 1: 1}, # M5: A00, A01
        {2: 1, 0: -1},# M6: A10, -A00
        {1: 1, 3: -1} # M7: A01, -A11
    ]
    
    R_coeffs = [
        {0: 1, 3: 1}, # M1: B00, B11
        {0: 1},       # M2: B00
        {1: 1, 3: -1},# M3: B01, -B11
        {2: 1, 0: -1},# M4: B10, -B00
        {3: 1},       # M5: B11
        {0: 1, 1: 1}, # M6: B00, B01
        {2: 1, 3: 1}  # M7: B10, B11
    ]
    
    # C block contributions: (block_row, block_col) -> coeff
    # C00: (0,0), C01: (0,1), C10: (1,0), C11: (1,1)
    E_coeffs = [
        {(0,0): 1, (1,1): 1}, # M1
        {(1,0): 1},           # M2
        {(0,1): 1, (1,1): -1},# M3
        {(0,0): 1, (1,0): 1}, # M4
        {(0,0): -1, (0,1): 1},# M5
        {(1,1): 1},           # M6
        {(0,0): 1, (1,1): 1}  # M7
    ]
    
    # Fill U, V, W for the 49 terms
    col = 0
    for r in range(7):
        for s in range(7):
            # Process term r, sub-term s
            # A part
            for i in range(2):
                for j in range(2):
                    u_val = U2[i*2+j][s]
                    if u_val != 0:
                        # Global A index: i is block row, j is block col
                        # Block index b = i*2 + j ? No, block 00 is (0,0), 01 is (0,1), 10 is (1,0), 11 is (1,1)
                        # My L_coeffs keys: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                        block_a = i*2 + j
                        c_a = L_coeffs[r].get(block_a, 0)
                        if c_a != 0:
                            # Local indices in the 2x2 block are i, j?
                            # Wait. The block A_{ij} is a 2x2 matrix.
                            # The local indices within that block are also 0,1.
                            # The global index I = 2*i + i_local.
                            # The U2 index was i_local * 2 + j_local.
                            # So for global I, J:
                            # I = 2*i + i_loc
                            # J = 2*j + j_loc
                            # U2 index is i_loc * 2 + j_loc.
                            # So we iterate i_loc, j_loc.
                            pass # Handled in loop below? No, I need to map global indices.
                            
            # It is easier to iterate over global indices for U, V, W directly.
            # For a fixed r, s:
            # Coeff of A[I, J]:
            # I = 2*i + i_loc, J = 2*j + j_loc
            # Block a = (i, j) -> idx i*2+j
            # c_a = L_coeffs[r].get(i*2+j, 0)
            # u_loc = U2[i_loc*2 + j_loc][s]
            # Total = c_a * u_loc
            
            # Coeff of B[J, K]:
            # J = 2*j + j_loc, K = 2*k + k_loc
            # Block b = (j, k) -> idx j*2+k
            # c_b = R_coeffs[r].get(j*2+k, 0)
            # v_loc = V2[j_loc*2 + k_loc][s]
            # Total = c_b * v_loc
            
            # Coeff of C[I, K]:
            # I = 2*i + i_loc, K = 2*k + k_loc
            # Block c = (i, k)
            # c_c = E_coeffs[r].get((i, k), 0)
            # w_loc = W2[k_loc*2 + i_loc][s] # Note: W2 index is k_loc * 2 + i_loc
            # Total = c_c * w_loc
            
            for I in range(4):
                for J in range(4):
                    i = I // 2
                    j = J // 2
                    i_loc = I % 2
                    j_loc = J % 2
                    block_a = i * 2 + j
                    c_a = L_coeffs[r].get(block_a, 0)
                    u_loc = U2[i_loc * 2 + j_loc][s]
                    val = c_a * u_loc
                    if val != 0:
                        U[I * m + J][col] = val
                        
            for J in range(4):
                for K in range(4):
                    j = J // 2
                    k = K // 2
                    j_loc = J % 2
                    k_loc = K % 2
                    block_b = j * 2 + k
                    c_b = R_coeffs[r].get(block_b, 0)
                    v_loc = V2[j_loc * 2 + k_loc][s]
                    val = c_b * v_loc
                    if val != 0:
                        V[J * p + K][col] = val
                        
            for I in range(4):
                for K in range(4):
                    i = I // 2
                    k = K // 2
                    i_loc = I % 2
                    k_loc = K % 2
                    block_c = (i, k)
                    c_c = E_coeffs[r].get(block_c, 0)
                    w_loc = W2[k_loc * 2 + i_loc][s]
                    val = c_c * w_loc
                    if val != 0:
                        W[K * n + I][col] = val
                            
            col += 1
            
    # --- Part 2: Matrix-Vector for 5th column ---
    # C[:, 4] = A B[:, 4]
    # c_i4 = sum_j a_ij b_j4
    # 16 terms
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
