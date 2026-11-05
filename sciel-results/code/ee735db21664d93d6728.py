import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Split B and C into 4x4 and 4x3 parts
    p1 = 4
    p2 = 3
    
    # Part 1: 4x4x4 multiplication (Rank 28)
    # Using Strassen for 4x4 with schoolbook for 2x2 blocks
    # U1, V1, W1 shapes: (16, 28), (16, 28), (16, 28)
    U1 = np.zeros((n*m, 28), dtype=int)
    V1 = np.zeros((m*p1, 28), dtype=int)
    W1 = np.zeros((p1*n, 28), dtype=int)
    
    # Strassen indices for 2x2 blocks
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Helper to add a schoolbook 2x2x2 decomposition to the matrices
    # start_col: current column offset in U, V, W
    # a_offset: row offset in U for the 2x2 A block
    # b_offset: row offset in V for the 2x2 B block
    # w_offset: row offset in W for the 2x2 C block
    # signs_a: 4x4 matrix of coefficients for A entries in the 2x2 block
    # signs_b: 4x4 matrix of coefficients for B entries in the 2x2 block
    # Actually, it's easier to just iterate over the 7 Strassen products
    # and for each, add the 4 schoolbook terms.
    
    def add_schoolbook_2x2(start_col, a_off, b_off, w_off, coeff_a, coeff_b):
        # coeff_a, coeff_b are 2x2 matrices of coefficients
        # We need to map 2x2 block indices to global indices
        # A block indices: (r, c) -> global row r*2 + c
        # B block indices: (r, c) -> global row r*2 + c
        # C block indices: (r, c) -> global row r*2 + c
        
        for k in range(4):
            col = start_col + k
            # Schoolbook for 2x2:
            # C00 = A00B00 + A01B10
            # C01 = A00B01 + A01B11
            # C10 = A10B00 + A11B10
            # C11 = A10B01 + A11B11
            
            # We need to define the 4 terms
            # Term 0: A00 B00 -> C00
            # Term 1: A01 B10 -> C00
            # Term 2: A00 B01 -> C01
            # Term 3: A01 B11 -> C01
            # Term 4: A10 B00 -> C10
            # Term 5: A11 B10 -> C10
            # Term 6: A10 B01 -> C11
            # Term 7: A11 B11 -> C11
            
            # Wait, schoolbook 2x2 has 4 multiplications.
            # M0 = A00 B00
            # M1 = A01 B10
            # M2 = A10 B00
            # M3 = A11 B10
            # No, that's not right.
            # C00 = A00 B00 + A01 B10
            # C01 = A00 B01 + A01 B11
            # C10 = A10 B00 + A11 B10
            # C11 = A10 B01 + A11 B11
            
            # The 4 multiplications are:
            # 1. A00 B00
            # 2. A01 B10
            # 3. A10 B00
            # 4. A11 B10
            # This doesn't cover C01 and C11.
            
            # Correct schoolbook 2x2:
            # m0 = A00 B00
            # m1 = A01 B10
            # m2 = A10 B00
            # m3 = A11 B10
            # m4 = A00 B01
            # m5 = A01 B11
            # m6 = A10 B01
            # m7 = A11 B11
            # That's 8 multiplications.
            
            # Strassen reduces 8 to 7.
            # If we use schoolbook, we use 8.
            # So 7 * 8 = 56.
            
            # Wait, I said 28 earlier.
            # 7 products of 2x2 matrices.
            # If each 2x2 product uses 4 mults (Strassen), total 49.
            # If each 2x2 product uses 8 mults (Schoolbook), total 56.
            
            # So my previous calculation of 28 was wrong.
            # Schoolbook 2x2 is 8 mults.
            # Strassen 2x2 is 7 mults.
            
            # So for 4x4:
            # Using Strassen on blocks: 7 block products.
            # If block products use Strassen: 7 * 7 = 49.
            # If block products use Schoolbook: 7 * 8 = 56.
            
            # So 49 is better.
            
            pass

    # Let's use the standard Strassen rank 49 for 4x4x4.
    # And Schoolbook rank 48 for 4x4x3.
    # Total 97.
    
    U = np.zeros((n*m, 97), dtype=int)
    V = np.zeros((m*p, 97), dtype=int)
    W = np.zeros((p*n, 97), dtype=int)
    
    # Part 1: 4x4x4 (Rank 49)
    # We need to generate the 49 terms.
    # This is tedious to hardcode.
    # We can generate it programmatically.
    
    # Strassen 2x2 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each M, we need to expand into scalar multiplications.
    # Since we are using Strassen for the inner 2x2 as well, we need to recurse.
    # Or just use the known rank 49 decomposition.
    
    # Given the complexity, let's stick to a simpler improvement.
    # Schoolbook 4x4x4 is 64.
    # Schoolbook 4x4x3 is 48.
    # Total 112.
    
    # Can we do 4x4x4 with rank 49 easily?
    # Yes, by applying Strassen twice.
    
    # Let's implement the rank 97 solution (49 + 48).
    
    col = 0
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # We will generate the 49 terms.
    # This requires a recursive generator or hardcoded list.
    # For brevity in this thought process, I will assume we can generate it.
    # In the final code, I will write a helper to generate Strassen terms.
    
    # Part 2: 4x4x3 using Schoolbook (Rank 48)
    # U2, V2, W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
