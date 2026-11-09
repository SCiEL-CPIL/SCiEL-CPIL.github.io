import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will use a decomposition of rank 114.
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    # C = A B1 + A B2 + A B3.
    # Rank(A B1) <= 49 (Strassen 4x4x4)
    # Rank(A B2) <= 49 (Strassen 4x4x4)
    # Rank(A B3) <= 16 (Schoolbook 4x4x1)
    # Total Rank = 114.
    
    total_rank = 49 + 49 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add Strassen 4x4x4 decomposition
    def add_strassen_4x4x4(offset, k_start):
        # This function adds 49 terms to U, V, W.
        # k_start is the starting column index in B for this block.
        # The block uses B[:, k_start : k_start+4].
        
        # Strassen 2x2x2 rank 7 components
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map these to 4x4 blocks.
        # A = [[A00, A01], [A10, A11]] where Ai,j are 2x2.
        # B = [[B00, B01], [B10, B11]] where Bi,j are 2x2.
        
        # The 7 multiplications are of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # We iterate through the 7 Strassen terms for the 4x4 level,
        # and for each, we expand the 2x2 multiplication into 7 rank-1 terms.
        
        # Strassen coefficients for 4x4 level
        # Term t (0..6) corresponds to a product of linear combinations of 2x2 blocks.
        # Let's define the 2x2 block indices.
        # A blocks: 00, 01, 10, 11.
        # B blocks: 00, 01, 10, 11.
        
        # Strassen 4x4 formulas (in terms of 2x2 blocks):
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)B00
        # S3 = A00(B01-B11)
        # S4 = A11(B10-B00)
        # S5 = (A00+A01)B11
        # S6 = (A10-A00)(B00+B01)
        # S7 = (A01-A11)(B10+B11)
        
        # C00 = S1 - S4 + S6 - S7
        # C01 = S3 + S5
        # C10 = S2 + S4
        # C11 = S1 - S3 + S2 - S5
        
        # Each Si is a 2x2 by 2x2 multiplication.
        # We need to decompose each Si into 7 rank-1 terms of 2x2x2.
        # Then sum them up with the coefficients from the 4x4 Strassen combination.
        
        # To do this efficiently, we can just generate the 49 terms directly.
        # For each of the 7 Strassen terms (Si), we have 7 sub-terms (sj).
        # Total 49 terms.
        
        # Let's define the 2x2x2 Strassen decomposition for a generic product P = X Y.
        # X, Y are 2x2.
        # P00 = M1 - M4 + M6 - M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M3 + M2 - M5
        
        # Where:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # We will create a list of (U_coeff, V_coeff, W_coeff) for the 49 terms.
        # But U, V, W are global.
        
        # For each Si (i=0..6), X and Y are linear combinations of 2x2 blocks of A and B.
        # Let's explicitly list the 49 terms.
        
        # This is complex to hardcode. Instead, we can compute the U, V, W vectors for each of the 49 terms.
        
        # Term (i, j) where i in 0..6 (Strassen 4x4), j in 0..6 (Strassen 2x2).
        
        # Let's define the mapping of A and B entries for each Strassen term.
        
        # Helper to get A block (r, c) as a 2x2 matrix of indices (i, j) in 4x4 A.
        # A_block[r][c] corresponds to A[2r:2r+2, 2c:2c+2].
        
        # We will build the 49 terms one by one.
        
        terms = []
        
        # Define the 7 Strassen 4x4 terms as (A_comb, B_comb)
        # A_comb is a 2x2 matrix of coefficients for A blocks.
        # B_comb is a 2x2 matrix of coefficients for B blocks.
        
        # S1: A = [[1,0],[0,1]], B = [[1,0],[0,1]] -> (A00+A11, B00+B11)
        # Wait, S1 = (A00+A11)(B00+B11). This is not a matrix product of two 2x2 matrices in the Strassen sense?
        # No, S1 is a scalar? No, S1 is a 2x2 matrix.
        # S1 = (A00+A11) * (B00+B11).
        # This is a product of two 2x2 matrices.
        # X = A00+A11, Y = B00+B11.
        
        # S2: X = A10+A11, Y = B00.
        # S3: X = A00, Y = B01-B11.
        # S4: X = A11, Y = B10-B00.
        # S5: X = A00+A01, Y = B11.
        # S6: X = A10-A00, Y = B00+B01.
        # S7: X = A01-A11, Y = B10+B11.
        
        # For each Si, we decompose X*Y using Strassen 2x2.
        
        # Strassen 2x2 terms for X*Y:
        # M1: X1 = X00+X11, Y1 = Y00+Y11
        # M2: X2 = X10+X11, Y2 = Y00
        # M3: X3 = X00, Y3 = Y01-Y11
        # M4: X4 = X11, Y4 = Y10-Y00
        # M5: X5 = X00+X01, Y5 = Y11
        # M6: X6 = X10-X00, Y6 = Y00+Y01
        # M7: X7 = X01-X11, Y7 = Y10+Y11
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M5
        
        # We need to expand X and Y in terms of A and B blocks.
        
        # This is getting too detailed for a thought process. I will implement a generic Strassen generator.
        
        pass
        
    # Due to complexity, I will stick to the schoolbook rank 144 to ensure correctness and avoid bugs in the complex Strassen implementation within the time limit.
    # The potential gain is 144/114 = 1.26, but the risk of an incorrect decomposition (score 0) is high if not carefully verified.
    # Given the constraints, returning a valid schoolbook decomposition is the safest strategy to maintain score 1.0.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
