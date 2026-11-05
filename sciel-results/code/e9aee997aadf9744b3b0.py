"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 6, 6, 6
    
    # Strassen decomposition for 6x6 with 3x3 blocks
    # Rank = 7 * 27 = 187
    
    rank = 187
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Block indexing helpers
    def a_idx(br, bc, ii, jj):
        return (br * 3 + ii) * 6 + (bc * 3 + jj)
    
    def b_idx(br, bc, ii, jj):
        return (br * 3 + ii) * 6 + (bc * 3 + jj)
    
    def w_idx(out_br, out_bc, ii, jj):
        # Following schoolbook: W[k*n+i] where i=out_row, k=out_col
        out_row = out_br * 3 + ii
        out_col = out_bc * 3 + jj
        return out_col * 6 + out_row
    
    # Strassen multiplications: (A_coeffs, B_coeffs, output_weights)
    # A_coeffs: {(block_row, block_col): coeff}
    # B_coeffs: {(block_row, block_col): coeff}
    # output_weights: {(out_block_row, out_block_col): coeff}
    
    strassen = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1},
         {(0,0):1, (1,1):1}),  # C00 += M1, C11 += M1
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1},
         {(1,0):1, (1,1):-1}),  # C10 += M2, C11 -= M2
        # M3 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1},
         {(0,1):1, (1,1):1}),  # C01 += M3, C11 += M3
        # M4 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1},
         {(0,0):1, (1,0):1}),  # C00 += M4, C10 += M4
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1},
         {(0,0):-1, (0,1):1}),  # C00 -= M5, C01 += M5
        # M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1},
         {(1,1):1}),  # C11 += M6
        # M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1},
         {(0,0):1}),  # C00 += M7
    ]
    
    col = 0
    for a_coeffs, b_coeffs, out_weights in strassen:
        if time.time() > deadline - 5:
            break
        for ii in range(3):
            for kk in range(3):
                for jj in range(3):
                    for (abr, abc), ac in a_coeffs.items():
                        for (bbr, bbc), bc in b_coeffs.items():
                            a_flat = a_idx(abr, abc, ii, kk)
                            b_flat = b_idx(bbr, bbc, kk, jj)
                            for (obr, obc), wc in out_weights.items():
                                w_flat = w_idx(obr, obc, ii, jj)
                                U[a_flat][col] = ac * bc * wc
                                V[b_flat][col] = 1
                                W[w_flat][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
