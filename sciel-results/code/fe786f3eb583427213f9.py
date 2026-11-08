import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use a 4+1 split
    # Indices 0-3 are the 4x4 block, index 4 is the 1x1 block
    
    terms = []
    
    # 1. 4x4 Block (Top-Left)
    # Use Strassen for 4x4
    # Strassen 4x4 is 49 terms if we use Strassen for the 2x2 sub-blocks too.
    # Let's generate the 49 terms.
    
    # Helper to get indices for a 2x2 block in a 4x4 matrix
    # Block (b_row, b_col) where b_row, b_col in {0,1}
    # Each block is 2x2.
    
    # Strassen 2x2 terms:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For 4x4, we apply this to the 2x2 blocks of 2x2 matrices.
    # This is effectively applying Strassen twice.
    
    # It's easier to just use the known 49-term decomposition or generate it.
    # Generating it programmatically is safer to avoid errors.
    
    # Let's define the 7 Strassen operations on 2x2 matrices.
    # Each operation produces a 2x2 matrix.
    # We need to expand each 2x2 matrix product into 4 scalar terms (if schoolbook) 
    # or 7 scalar terms (if Strassen).
    # Since we want rank 49, we use Strassen for the inner 2x2s as well.
    
    # This is getting complex to code from scratch in 150s.
    # Simpler: Use Schoolbook for the 4x4 block?
    # Schoolbook 4x4 is 64 terms.
    # Total Rank = 64 (4x4) + 1 (1x1) + 48 (cross) = 113.
    # 113 is better than 125, but 98 is better.
    
    # Can we easily get 98?
    # Yes, by using Strassen for the 4x4.
    
    # Let's try to implement the 4x4 Strassen generation.
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        terms.append((u_idx, v_idx, w_idx, val))

    # 4x4 Strassen
    # We map the 4x4 indices (0..3) to 2x2 blocks (0..1) and within-block (0..1).
    # i = 2*bi + ri, j = 2*bj + rj
    
    # Strassen terms for 4x4:
    # There are 7 "block" terms. Each block term is a product of two 2x2 matrices.
    # Each 2x2 matrix product can be done with 7 scalar terms (Strassen).
    # So 7 * 7 = 49 terms.
    
    # We need the definitions of the 7 block terms.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # Let B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to expand each Mk into 7 scalar terms.
    # This requires a function that takes two 2x2 matrices (defined by linear combos of A and B)
    # and returns the 7 Strassen scalar terms.
    
    # This is too much code for the time limit.
    # Let's stick to **Schoolbook for 4x4** (64 terms) for simplicity and reliability.
    # Rank 113 is still an improvement.
    
    # Wait, I can use **Strassen for 2x2** on the 2x2 sub-blocks of the 4x4?
    # If I just use schoolbook for the 4x4, I get 64.
    
    # Let's try to implement the **2+3 split** with Strassen 2x2 and Schoolbook 3x3?
    # 2x2 Strassen: 7 terms.
    # 3x3 Schoolbook: 27 terms.
    # Cross terms:
    # 2x3 to 2x2: 12
    # 3x2 to 3x3: 18
    # 2x2 to 2x3: 12
    # 2x3 to 3x3: 18
    # 3x2 to 2x3: 18
    # 3x3 to 3x2: 18
    # Total Cross: 96.
    # Total: 7 + 27 + 96 = 130. Worse than 125.
    
    # So the **4+1 split with Schoolbook 4x4** is the best simple structural change.
    # Rank 113.
    
    terms = []
    
    # 4x4 Block (Indices 0-3)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # U index: i*5 + j
                # V index: j*5 + k
                # W index: k*5 + i
                terms.append((i*5+j, j*5+k, k*5+i, 1))
                
    # 1x1 Block (Index 4)
    terms.append((4*5+4, 4*5+4, 4*5+4, 1))
    
    # Cross Terms
    # C44 = A44 B44 + A41 B14
    # We already did A44 B44.
    # Add A41 B14
    # A41 is row 0-3, col 4. B14 is row 4, col 0-3.
    for i in range(4):
        for k in range(4):
            # j=4
            terms.append((i*5+4, 4*5+k, k*5+i, 1))
            
    # C41 = A44 B41 + A41 B11
    # A44 B41: i in 0-3, j in 0-3, k=4
    for i in range(4):
        for j in range(4):
            terms.append((i*5+j, j*5+4, 4*5+i, 1))
            
    # A41 B11: i in 0-3, j=4, k=4
    for i in range(4):
        terms.append((i*5+4, 4*5+4, 4*5+i, 1))
        
    # C14 = A14 B44 + A11 B14
    # A14 B44: i=4, j in 0-3, k in 0-3
    for j in range(4):
        for k in range(4):
            terms.append((4*5+j, j*5+k, k*5+4, 1))
            
    # A11 B14: i=4, j=4, k in 0-3
    for k in range(4):
        terms.append((4*5+4, 4*5+k, k*5+4, 1))
        
    # C11 = A14 B41 + A11 B11
    # A14 B41: i=4, j in 0-3, k=4
    for j in range(4):
        terms.append((4*5+j, j*5+4, 4*5+4, 1))
        
    # A11 B11: Already added (1x1 block)
    
    # Total terms: 64 + 1 + 16 + 16 + 4 + 16 + 4 + 4 = 125?
    # Let's recount.
    # 4x4: 64
    # 1x1: 1
    # A41 B14: 16
    # A44 B41: 16
    # A41 B11: 4
    # A14 B44: 16
    # A11 B14: 4
    # A14 B41: 4
    # Sum: 64 + 1 + 16 + 16 + 4 + 16 + 4 + 4 = 125.
    
    # It's still 125!
    # Why? Because schoolbook 4x4 is 64, and the cross terms sum to 61.
    # 64 + 1 + 60 = 125.
    # The split didn't help because we used schoolbook for the 4x4.
    
    # We MUST use Strassen for the 4x4 to save rank.
    # 49 terms for 4x4.
    # 49 + 1 + 60 = 110.
    
    # I will implement the 4x4 Strassen.
    
    # To save time, I will hardcode the 49 terms for 4x4 Strassen?
    # No, I will generate them using a recursive Strassen function.
    
    # Let's write a helper to generate Strassen terms for 2x2.
    # Then apply it to 4x4.
    
    return {"U": [], "V": [], "W": []} # Placeholder
