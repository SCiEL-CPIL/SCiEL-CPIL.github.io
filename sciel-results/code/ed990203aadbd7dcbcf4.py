import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Tensor dimensions
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    # Strassen decomposition for 2x2 matrix multiplication.
    # We treat A as 2x2 scalar matrix.
    # We treat B as 2x2 matrix over the ring R = F^5 (component-wise addition/multiplication).
    # B_block[0,0] = B[:, 0:5] (vector of 5)
    # B_block[0,1] = B[:, 5:10] (vector of 5)
    # B_block[1,0] = B[1, 0:5] ... wait, B is 2x10.
    # Let's be precise with indices.
    # A is 2x2. Indices: A[i, j] -> index i*m + j = i*2 + j.
    # B is 2x10. Indices: B[j, k] -> index j*p + k = j*10 + k.
    # C is 2x10. Indices: C[i, k] -> index k*n + i = k*2 + i.
    
    # Strassen Products for C = A * B:
    # Let A = [[a00, a01], [a10, a11]]
    # Let B be split into B_L (cols 0-4) and B_R (cols 5-9).
    # B_L = [[b00_L, b01_L], [b10_L, b11_L]] where b_ij_L is a 5-vector.
    # B_R = [[b00_R, b01_R], [b10_R, b11_R]] where b_ij_R is a 5-vector.
    
    # Actually, standard Strassen applies to the 2x2 block structure.
    # C_L = A * B_L (2x5 result)
    # C_R = A * B_R (2x5 result)
    
    # Strassen on the 2x2 "blocks" (where blocks are 1x5 vectors):
    # M1 = (A00 + A11) * (B00_L + B11_R)  -- Wait, standard Strassen mixes blocks.
    # Let's stick to the standard 7 products for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Here, B00, B01, B10, B11 are the 2x2 blocks of B.
    # Since B is 2x10, we can view it as a 2x2 matrix of 1x5 vectors.
    # B[0,0] = B[0, 0:5]
    # B[0,1] = B[0, 5:10]
    # B[1,0] = B[1, 0:5]
    # B[1,1] = B[1, 5:10]
    
    # The products M1..M7 are computed in the ring of 1x5 vectors.
    # This means M1 is a 1x5 vector, computed as (scalar) * (vector + vector).
    # Specifically, M1[t] = (A00+A11) * (B[0,0:5][t] + B[1,5:10][t])
    
    # The reconstruction of C:
    # C[0,0] (vector 0:5) = M1 + M4 - M5 + M7
    # C[0,1] (vector 5:10) = M3 + M5
    # C[1,0] (vector 0:5) = M2 + M4
    # C[1,1] (vector 5:10) = M1 - M2 + M3 + M6
    
    # Note: C[0,0] refers to C[0, 0:5], which maps to indices 0, 2, 4, 6, 8 in W (k*n+i).
    # C[0,1] refers to C[0, 5:10], which maps to indices 10, 12, 14, 16, 18 in W.
    # C[1,0] refers to C[1, 0:5], which maps to indices 1, 3, 5, 7, 9 in W.
    # C[1,1] refers to C[1, 5:10], which maps to indices 11, 13, 15, 17, 19 in W.
    
    # Let's define the Strassen terms programmatically.
    # Each term has:
    # 1. A linear combination of A entries (for U)
    # 2. A linear combination of B entries (for V)
    # 3. A linear combination of C entries (for W)
    
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices: 
    #   B[0,0:5] -> 0..4
    #   B[0,5:10] -> 5..9
    #   B[1,0:5] -> 10..14
    #   B[1,5:10] -> 15..19
    # C indices (W):
    #   C[0,0:5] -> 0, 2, 4, 6, 8
    #   C[0,5:10] -> 10, 12, 14, 16, 18
    #   C[1,0:5] -> 1, 3, 5, 7, 9
    #   C[1,5:10] -> 11, 13, 15, 17, 19
    
    # Define the 7 Strassen products
    # Format: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: list of (index, coeff)
    # B_coeffs: list of (block_row, block_col, coeff) -> we will expand to specific indices later
    # C_coeffs: list of (block_row, block_col, coeff) -> we will expand to specific indices later
    
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11)
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)] # Wait, C coeffs are for reconstruction
        },
        # M2 = (A10 + A11) * B00
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 0, 1)],
            'C': [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, -1)]
        },
        # M3 = A00 * (B01 - B11)
        {
            'A': [(0, 1)],
            'B': [(0, 1, 1), (1, 1, -1)],
            'C': [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]
        },
        # M4 = A11 * (B10 - B00)
        {
            'A': [(3, 1)],
            'B': [(1, 0, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0)]
        },
        # M5 = (A00 + A01) * B11
        {
            'A': [(0, 1), (1, 1)],
            'B': [(1, 1, 1)],
            'C': [(0, 0, -1), (0, 1, 1), (1, 0, 0), (1, 1, 0)]
        },
        # M6 = (A10 - A00) * (B00 + B01)
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'C': [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        },
        # M7 = (A01 - A11) * (B10 + B11)
        {
            'A': [(1, 1), (3, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
        }
    ]
    
    # Verify Strassen reconstruction logic:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check the C coeffs defined above against this:
    # M1: C00:+1, C11:+1. (Matches M1 in C00 and C11)
    # M2: C10:+1, C11:-1. (Matches M2 in C10 and -M2 in C11)
    # M3: C01:+1, C11:+1. (Matches M3 in C01 and C11)
    # M4: C00:+1, C10:+1. (Matches M4 in C00 and C10)
    # M5: C00:-1, C01:+1. (Matches -M5 in C00 and M5 in C01)
    # M6: C11:+1. (Matches M6 in C11)
    # M7: C00:+1. (Matches M7 in C00)
    
    # Summing for C00: M1(+1) + M4(+1) + M5(-1) + M7(+1) = 1+1-1+1 = 2? 
    # Wait. Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # My M5 def: (A00+A01)*B11.
    # My M7 def: (A01-A11)*(B10+B11).
    # Let's re-verify standard Strassen formulas.
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
    
    # My C coeffs in the list above:
    # Term 0 (M1): C00:+1, C11:+1. Correct.
    # Term 1 (M2): C10:+1, C11:-1. Correct.
    # Term 2 (M3): C01:+1, C11:+1. Correct.
    # Term 3 (M4): C00:+1, C10:+1. Correct.
    # Term 4 (M5): C00:-1, C01:+1. Correct.
    # Term 5 (M6): C11:+1. Correct.
    # Term 6 (M7): C00:+1. Correct.
    
    # The logic holds.
    
    rank = 35
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    # Block size is 5
    block_size = 5
    
    for term in strassen_terms:
        a_coeffs = term['A']
        b_coeffs = term['B']
        c_coeffs = term['C']
        
        for t in range(block_size):
            # Fill U
            for idx, coeff in a_coeffs:
                U[idx][col] = coeff
            
            # Fill V
            for b_row, b_col, coeff in b_coeffs:
                # b_row is 0 or 1
                # b_col is 0 or 1 (left or right half)
                # Actual index in V (flattened B):
                # B is 2x10. Index = row * 10 + col.
                # If b_col=0, actual cols are 0..4. If b_col=1, actual cols are 5..9.
                actual_b_col = b_col * block_size + t
                v_idx = b_row * p + actual_b_col
                V[v_idx][col] = coeff
            
            # Fill W
            for c_row, c_col, coeff in c_coeffs:
                # c_row is 0 or 1
                # c_col is 0 or 1 (left or right half)
                # Actual index in W (flattened C):
                # C is 2x10. Index = k * 2 + i.
                # Here c_row is i (0 or 1), c_col is block of k.
                # If c_col=0, actual k are 0..4. If c_col=1, actual k are 5..9.
                actual_c_k = c_col * block_size + t
                w_idx = actual_c_k * n + c_row
                W[w_idx][col] = coeff
            
            col += 1
    
    # Optional: Verification (removed for speed, but logically sound)
    # We can quickly verify one entry to be sure.
    # C[0,0] (k=0, i=0) should be A00*B00 + A01*B10
    # W[0] gets contributions from terms where c_row=0, c_col=0, t=0.
    # Terms with c_row=0, c_col=0: M1(+1), M4(+1), M5(-1), M7(+1).
    # M1: A=(0,1),(3,1); B=(0,0,1),(1,1,1).
    #   U: A00+A11. V: B00+B11.
    #   Contribution: (A00+A11)*(B00+B11)
    # M4: A=(3,1); B=(1,0,1),(0,0,-1).
    #   U: A11. V: B10-B00.
    #   Contribution: A11*(B10-B00)
    # M5: A=(0,1),(1,1); B=(1,1,1).
    #   U: A00+A01. V: B11.
    #   Contribution: -(A00+A01)*B11
    # M7: A=(1,1),(3,-1); B=(1,0,1),(1,1,1).
    #   U: A01-A11. V: B10+B11.
    #   Contribution: (A01-A11)*(B10+B11)
    
    # Sum:
    # (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # Cancel terms:
    # A11B00 - A11B00 = 0
    # A00B11 - A00B11 = 0
    # A11B11 - A11B11 = 0
    # -A01B11 + A01B11 = 0
    # A11B10 - A11B10 = 0
    # Remaining:
    # A00B00 + A01B10
    # This matches C[0,0] = A00*B[0,0] + A01*B[1,0].
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
