"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Decompose 2x2 * 2x10 matrix multiplication using the block-diagonal structure.
    The 2x10 matrix can be viewed as 5 independent 2x2 blocks (columns 0-1, 2-3, etc.).
    Each 2x2 * 2x2 multiplication can be done with Strassen's rank-7 algorithm.
    Total rank = 5 * 7 = 35.
    
    Output convention: W[k*n+i, r] where n=2, p=10, m=2.
    So W has shape (p*n, rank) = (20, 35).
    U has shape (n*m, rank) = (4, 35).
    V has shape (m*p, rank) = (20, 35).
    """
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Number of 2x2 blocks in the p dimension: p // 2 = 5
    num_blocks = p // 2  # 5
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block  # 35
    
    U = [[0] * total_rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * total_rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * total_rank for _ in range(p * n)]  # (20, 35)
    
    # Strassen's 7 terms for 2x2 matrix multiplication:
    # C11 = A11*(B11+B22) - M5 + M3 - M7  ... wait, let me use the standard Strassen formulas.
    #
    # For C = A * B where A and B are 2x2:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen's 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # Then:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block b (0 to 4):
    # The block corresponds to columns 2*b and 2*b+1 of the B matrix.
    # A is 2x2, B_block is 2x2 (rows 0,1 of B, columns 2*b, 2*b+1)
    # C_block is 2x2 (rows 0,1 of C, columns 2*b, 2*b+1)
    
    # Indexing for U (A matrix, shape n*m = 4):
    # A[i,j] is at U[i*m + j] = U[i*2 + j]
    # A11 = A[0,0] -> U[0]
    # A12 = A[0,1] -> U[1]
    # A21 = A[1,0] -> U[2]
    # A22 = A[1,1] -> U[3]
    
    # Indexing for V (B matrix, shape m*p = 20):
    # B[j,k] is at V[j*p + k] = V[j*10 + k]
    # For block b, columns are 2*b and 2*b+1
    # B11 = B[0, 2*b] -> V[0*10 + 2*b] = V[2*b]
    # B12 = B[0, 2*b+1] -> V[2*b+1]
    # B21 = B[1, 2*b] -> V[1*10 + 2*b] = V[10 + 2*b]
    # B22 = B[1, 2*b+1] -> V[10 + 2*b + 1] = V[10 + 2*b + 1]
    
    # Indexing for W (C matrix output, shape p*n = 20):
    # C[k,i] is at W[k*n + i] = W[k*2 + i]
    # For block b, columns are 2*b and 2*b+1
    # C11 = C[0, 2*b] -> W[2*b*2 + 0] = W[4*b]
    # C12 = C[0, 2*b+1] -> W[2*b*2 + 1] = W[4*b + 1]
    # C21 = C[1, 2*b] -> W[(2*b+1)*2 + 0] = W[4*b + 2]
    # C22 = C[1, 2*b+1] -> W[(2*b+1)*2 + 1] = W[4*b + 3]
    
    # Strassen terms and their contributions:
    # Each term t (0..6) has:
    # - A coefficients: which U rows get which values
    # - B coefficients: which V rows get which values
    # - C contributions: which W rows get +1 or -1
    
    # Term definitions:
    # M1 = (A11 + A22) * (B11 + B22)
    #   A: U[0] += 1, U[3] += 1
    #   B: V[2*b] += 1, V[10+2*b+1] += 1
    #   C: C11 += 1, C22 += 1
    #   W: W[4*b] += 1, W[4*b+3] += 1
    
    # M2 = (A21 + A22) * B11
    #   A: U[2] += 1, U[3] += 1
    #   B: V[2*b] += 1
    #   C: C21 += 1, C22 -= 1
    #   W: W[4*b+2] += 1, W[4*b+3] -= 1
    
    # M3 = A11 * (B12 - B22)
    #   A: U[0] += 1
    #   B: V[2*b+1] += 1, V[10+2*b+1] -= 1
    #   C: C12 += 1, C22 += 1
    #   W: W[4*b+1] += 1, W[4*b+3] += 1
    
    # M4 = A22 * (B21 - B11)
    #   A: U[3] += 1
    #   B: V[10+2*b] += 1, V[2*b] -= 1
    #   C: C11 += 1, C21 += 1
    #   W: W[4*b] += 1, W[4*b+2] += 1
    
    # M5 = (A11 + A12) * B22
    #   A: U[0] += 1, U[1] += 1
    #   B: V[10+2*b+1] += 1
    #   C: C11 -= 1
    #   W: W[4*b] -= 1
    
    # M6 = (A21 - A11) * (B11 + B12)
    #   A: U[2] += 1, U[0] -= 1
    #   B: V[2*b] += 1, V[2*b+1] += 1
    #   C: C22 += 1
    #   W: W[4*b+3] += 1
    
    # M7 = (A12 - A22) * (B21 + B22)
    #   A: U[1] += 1, U[3] -= 1
    #   B: V[10+2*b] += 1, V[10+2*b+1] += 1
    #   C: C11 += 1
    #   W: W[4*b] += 1
    
    # Let's define the 7 terms as tuples of (U_rows_with_coeffs, V_rows_with_coeffs, W_rows_with_coeffs)
    # Each is a list of (row_index, coefficient)
    
    for b in range(num_blocks):
        col_offset = 2 * b  # column offset in B and C
        base_rank = b * rank_per_block  # base rank offset
        
        # Define the 7 Strassen terms
        # Each term: (list of (U_row, coeff), list of (V_row, coeff), list of (W_row, coeff))
        
        terms = [
            # M1 = (A11 + A22) * (B11 + B22)
            ([(0, 1), (3, 1)],
             [(col_offset, 1), (10 + col_offset + 1, 1)],
             [(4*b, 1), (4*b+3, 1)]),
            
            # M2 = (A21 + A22) * B11
            ([(2, 1), (3, 1)],
             [(col_offset, 1)],
             [(4*b+2, 1), (4*b+3, -1)]),
            
            # M3 = A11 * (B12 - B22)
            ([(0, 1)],
             [(col_offset+1, 1), (10 + col_offset + 1, -1)],
             [(4*b+1, 1), (4*b+3, 1)]),
            
            # M4 = A22 * (B21 - B11)
            ([(3, 1)],
             [(10 + col_offset, 1), (col_offset, -1)],
             [(4*b, 1), (4*b+2, 1)]),
            
            # M5 = (A11 + A12) * B22
            ([(0, 1), (1, 1)],
             [(10 + col_offset + 1, 1)],
             [(4*b, -1)]),
            
            # M6 = (A21 - A11) * (B11 + B12)
            ([(2, 1), (0, -1)],
             [(col_offset, 1), (col_offset+1, 1)],
             [(4*b+3, 1)]),
            
            # M7 = (A12 - A22) * (B21 + B22)
            ([(1, 1), (3, -1)],
             [(10 + col_offset, 1), (10 + col_offset + 1, 1)],
             [(4*b, 1)]),
        ]
        
        for t_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(terms):
            r = base_rank + t_idx
            for (row, coeff) in u_coeffs:
                U[row][r] = coeff
            for (row, coeff) in v_coeffs:
                V[row][r] = coeff
            for (row, coeff) in w_coeffs:
                W[row][r] = coeff
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
