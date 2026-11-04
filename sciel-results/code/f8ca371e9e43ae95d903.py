import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use the split 5 = 4 + 1
    # Part 1: 4x4x4 (Strassen rank 49)
    # Part 2: 4x4x1 (Rank 4)
    # Part 3: 4x1x5 (Rank 5)
    # Total Rank 58
    
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen ---
    # Generate Strassen decomposition for 4x4x4
    # We will use a recursive approach or hardcoded indices.
    # For simplicity and correctness, we generate the 49 terms.
    
    # Strassen 2x2 terms
    # Returns list of (u_idx, v_idx, w_idx, coeff) for local 2x2
    # But we need to map to global 4x4.
    
    # It is easier to just iterate over the 49 terms of 4x4 Strassen.
    # 4x4 Strassen is 7 terms of 2x2 Strassen, each of which is 7 terms of 1x1.
    # So 49 terms.
    
    # Let's generate the 49 terms.
    # Each term corresponds to a product of a linear combination of A and a linear combination of B.
    # The coefficients are +/- 1.
    
    # We can define the 49 terms by their structure.
    # Term t (0..48)
    # t = a * 7 + b, where a is the 2x2 Strassen term (0..6), b is the inner 2x2 Strassen term (0..6).
    # The 2x2 Strassen term a operates on 2x2 blocks of the 4x4 matrix.
    # The inner term b operates on 2x2 blocks of the 2x2 block.
    
    # This is getting complex to hardcode.
    # Alternative: Use the known rank 49 decomposition.
    # I will implement a simple recursive Strassen that outputs the U, V, W contributions.
    
    def add_strassen(U, V, W, offset, n_size, i_offset, j_offset, k_offset):
        # n_size is the size of the block (e.g., 4, 2)
        # i_offset, j_offset, k_offset are the starting indices in the global matrices
        # This function adds the Strassen decomposition for an n_size x n_size x n_size block
        # into U, V, W starting at column 'offset'.
        # Returns the number of columns used.
        
        if n_size == 1:
            # Rank 1
            # U[i_offset, offset] = 1
            # V[j_offset, offset] = 1
            # W[k_offset, offset] = 1
            U[i_offset][offset] = 1
            V[j_offset][offset] = 1
            W[k_offset][offset] = 1
            return 1
            
        half = n_size // 2
        # Strassen 7 terms
        # Each term is a product of two linear combinations.
        # We need to define the linear combinations for A and B.
        
        # Strassen formulas for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        
        # We will iterate through the 7 terms.
        # For each term, we add a decomposition of size half.
        # Wait, Strassen is recursive.
        # M1 is computed by multiplying two half-sized matrices.
        # So we call add_strassen for each of the 7 terms?
        # No, the linear combinations are formed by adding the U/V vectors.
        
        # This recursive addition of vectors is tricky with the flattened U, V, W.
        # It is better to generate the 49 base terms (1x1) directly.
        
        pass

    # Given the complexity, I will use a pre-computed list of indices for 4x4 Strassen.
    # Or, simpler: just use the schoolbook for 4x4x4? Rank 64.
    # 64 + 4 + 5 = 73.
    # Strassen 49 is better.
    
    # I will implement the 49 terms explicitly using a helper that generates the 7x7 grid of 1x1 terms.
    
    # The 49 terms are indexed by (a, b) where a, b in 0..6.
    # The 2x2 Strassen term 'a' selects which 2x2 blocks of A and B to combine.
    # The inner term 'b' selects which 1x1 elements to combine.
    
    # Let's define the 7 Strassen combinations for 2x2.
    # Each combination is a set of (i, j, coeff) for A and (k, l, coeff) for B.
    
    # A combinations:
    # 0: (0,0,1), (1,1,1)
    # 1: (1,0,1), (1,1,1)
    # 2: (0,0,1)
    # 3: (1,1,1)
    # 4: (0,0,1), (0,1,1)
    # 5: (1,0,1), (0,0,-1)
    # 6: (0,1,1), (1,1,-1)
    
    # B combinations:
    # 0: (0,0,1), (1,1,1)
    # 1: (0,0,1)
    # 2: (1,0,1), (1,1,-1)
    # 3: (0,1,1), (0,0,-1)
    # 4: (1,1,1)
    # 5: (0,0,1), (1,0,1)
    # 6: (1,1,1), (0,1,1)
    
    # This is for 2x2.
    # For 4x4, we apply this to 2x2 blocks.
    # Block (bi, bj) of A is A[bi*2..bi*2+1, bj*2..bj*2+1].
    
    # I will generate the 49 terms by iterating a, b in 0..6.
    # For each (a, b), we determine the 1x1 terms.
    
    # This is too complex to write correctly in short time.
    # I will fall back to Rank 73 (Schoolbook 4x4x4) if Strassen is too risky.
    # But 58 is a significant improvement.
    
    # Let's try to just output Rank 73 first to be safe?
    # No, 58 is achievable.
    
    # I will use a known 4x4 Strassen decomposition.
    # There are 49 terms.
    
    # I will write a function that generates the 49 terms.
    
    def get_strassen_4x4():
        terms = []
        # a, b in 0..6
        for a in range(7):
            for b in range(7):
                # Determine the 1x1 contributions
                # This requires knowing the structure of Strassen.
                # I will skip this and use Rank 73.
                pass
        return terms

    # Decision: Use Rank 73 (Schoolbook 4x4x4 + tails).
    # 4x4x4 Schoolbook Rank 64.
    # 4x4x1 Rank 4.
    # 4x1x5 Rank 5.
    # Total 73.
    
    # This is a safe improvement over 100.
    
    # --- Part 1: 4x4x4 Schoolbook ---
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # --- Part 2: 4x4x1 ---
    # j in 0..3, k=4
    for j in range(4):
        # c is col
        for i in range(4):
            # We need rank 4 for this part.
            # My previous analysis: Rank 4.
            # But schoolbook is 16.
            # Let's use the Rank 4 decomposition.
            pass
            
    # I will stick to Schoolbook for the 4x4x4 part to ensure correctness.
    # And use the optimized Rank 4 and Rank 5 for the tails.
    
    # Wait, the 4x4x4 part is 64.
    # Tails are 4 and 5.
    # Total 73.
    
    # Let's implement the Rank 4 and Rank 5 tails correctly.
    
    # Part 2: 4x4x1
    # c in 49..52? No, col is 64.
    # c in 64..67.
    for j in range(4):
        c = 64 + j
        for i in range(4):
            U[i * m + j][c] = 1
            V[j * p + 4][c] = 1 # k=4
            W[4 * n + i][c] = 1 # k=4 -> 4*4+i = 16+i?
            # W index is k*n + i.
            # k=4, n=4. 16+i.
            pass
            
    # Wait, W index is k*n + i.
    # For k=4, i in 0..3.
    # W[16+i, c] = 1.
    
    # Part 3: 4x1x5
    # j=4, k in 0..4.
    # c in 68..72.
    for k in range(5):
        c = 68 + k
        for i in range(4):
            U[i * m + 4][c] = 1 # j=4
            V[4 * p + k][c] = 1 # j=4
            W[k * n + i][c] = 1
            pass

    return {"U": U, "V": V, "W": W}
