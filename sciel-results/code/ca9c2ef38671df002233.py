"""Strassen-based rank-189 decomposition for 6x6x6 matrix multiplication."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's 2x2 block decomposition on 3x3 blocks.
    # 7 block multiplications of 3x3 matrices, each schoolbook (rank 27).
    # Total rank = 7 * 27 = 189
    
    # U[i*6+j, c], V[j*6+k, c], W[k*6+i, c]
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_c U[i*6+j,c]*V[j*6+k,c]*W[k*6+i,c]
    
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen terms:
    # Each term has A_blocks, B_blocks, and C_contrib
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    # C_contrib: list of (block_row, block_col, coeff)
    
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11) -> C00(+1), C11(+1)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A10 + A11) * B00 -> C10(+1), C11(-1)
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3 = A00 * (B01 - B11) -> C01(+1), C11(+1)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4 = A11 * (B10 - B00) -> C10(+1), C00(+1)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(1,0,1), (0,0,1)]),
        # M5 = (A00 + A01) * B11 -> C00(-1), C01(+1)
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6 = (A10 - A00) * (B00 + B01) -> C11(+1)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A01 - A11) * (B10 + B11) -> C00(+1)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    for A_blocks, B_blocks, C_contrib in strassen_terms:
        # 3x3 schoolbook: for each i,j,k in 0..2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for (abr, abc, ca) in A_blocks:
                        for (bbr, bbc, cb) in B_blocks:
                            coeff = ca * cb
                            if coeff == 0:
                                continue
                            
                            a_row = abr * 3 + i
                            a_col = abc * 3 + j
                            b_row = bbr * 3 + j
                            b_col = bbc * 3 + k
                            
                            U_idx = a_row * 6 + a_col
                            V_idx = b_row * 6 + b_col
                            
                            U[U_idx][col] += coeff
                            V[V_idx][col] += coeff
                            
                            for (cbr, cbc, cc) in C_contrib:
                                c_out_row = cbr * 3 + i
                                c_out_col = cbc * 3 + k
                                W_idx = c_out_col * 6 + c_out_row
                                W[W_idx][col] += coeff * cc
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
