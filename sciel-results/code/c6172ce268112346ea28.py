import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    # Use 3x3 block decomposition where each block is 2x2.
    # For each output block C_ik (i,k in 0..2), compute sum_j A_ij * B_jk
    # where each product is a 2x2 matrix multiplication using Strassen (rank 7).
    # Total rank: 9 * 3 * 7 = 189.
    
    nb = 3  # number of blocks per dimension
    bs = 2  # block size (2x2 submatrices)
    
    rank = nb * nb * (nb * 7)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication
    # For X, Y (2x2 matrices), C = X*Y:
    # M0 = (X00 + X11) * (Y00 + Y11)
    # M1 = (X10 + X11) * Y00
    # M2 = X00 * (Y01 - Y11)
    # M3 = X11 * (Y10 - Y00)
    # M4 = (X00 + X01) * Y11
    # M5 = (X10 - X00) * (Y00 + Y01)
    # M6 = (X01 - X11) * (Y10 + Y11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Define Strassen terms as (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict mapping (row, col) within 2x2 block to coefficient
    # B_coeffs: dict mapping (row, col) within 2x2 block to coefficient  
    # C_coeffs: dict mapping (row, col) within 2x2 block to coefficient
    
    strassen_terms = [
        # M0
        {'A': {(0,0): 1.0, (1,1): 1.0}, 
         'B': {(0,0): 1.0, (1,1): 1.0},
         'C': {(0,0): 1.0, (1,1): 1.0}},
        # M1
        {'A': {(1,0): 1.0, (1,1): 1.0},
         'B': {(0,0): 1.0},
         'C': {(1,0): 1.0, (1,1): -1.0}},
        # M2
        {'A': {(0,0): 1.0},
         'B': {(0,1): 1.0, (1,1): -1.0},
         'C': {(0,1): 1.0, (1,1): 1.0}},
        # M3
        {'A': {(1,1): 1.0},
         'B': {(1,0): 1.0, (0,0): -1.0},
         'C': {(0,0): 1.0, (1,0): 1.0}},
        # M4
        {'A': {(0,0): 1.0, (0,1): 1.0},
         'B': {(1,1): 1.0},
         'C': {(0,0): -1.0, (0,1): 1.0}},
        # M5
        {'A': {(1,0): 1.0, (0,0): -1.0},
         'B': {(0,0): 1.0, (0,1): 1.0},
         'C': {(1,1): 1.0}},
        # M6
        {'A': {(0,1): 1.0, (1,1): -1.0},
         'B': {(1,0): 1.0, (1,1): 1.0},
         'C': {(0,0): 1.0}},
    ]
    
    col = 0
    for i_block in range(nb):
        for k_block in range(nb):
            for j_block in range(nb):
                # Compute contribution of A[i_block, j_block] * B[j_block, k_block] to C[i_block, k_block]
                for s in range(7):
                    term = strassen_terms[s]
                    A_coeffs = term['A']
                    B_coeffs = term['B']
                    C_coeffs = term['C']
                    
                    for u in range(bs):
                        for v in range(bs):
                            # U factor: contribution from A_block[u, v]
                            for (ar, ac), coeff_a in A_coeffs.items():
                                i_glob = i_block * bs + ar
                                k_glob = j_block * bs + ac
                                # But we need to map (u,v) to the actual position
                                # A_block element at (ar, ac) corresponds to global position
                                # row: i_block*bs + ar, col: j_block*bs + ac
                                # But u and v are the indices within the 2x2 block
                                # Actually, for each (u,v) in the 2x2 block:
                                # A element at block position (u,v) is at global (i_block*bs+u, j_block*bs+v)
                                # The Strassen term uses specific (ar,ac) positions
                                # So for this (u,v), we add coeff_a to U if (ar,ac) == (u,v)
                                if (ar, ac) == (u, v):
                                    i_glob = i_block * bs + u
                                    k_glob = j_block * bs + v
                                    u_row = i_glob * m + k_glob
                                    U[u_row][col] += coeff_a
                            
                            # V factor: contribution from B_block[u, v]
                            for (br, bc), coeff_b in B_coeffs.items():
                                if (br, bc) == (u, v):
                                    k_glob = j_block * bs + u
                                    j_glob = k_block * bs + v
                                    v_row = k_glob * p + j_glob
                                    V[v_row][col] += coeff_b
                            
                            # W factor: contribution to C_block from this Strassen term
                            for (cr, cc), coeff_c in C_coeffs.items():
                                if (cr, cc) == (u, v):
                                    i_glob = i_block * bs + u
                                    j_glob = k_block * bs + v
                                    # W index: W[k*n + i, r] where k is col of C, i is row of C
                                    w_row = j_glob * n + i_glob
                                    W[w_row][col] += coeff_c
                    
                    col += 1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
