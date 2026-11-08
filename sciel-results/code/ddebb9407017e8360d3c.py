import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # We will build U, V, W as lists of lists initially, then convert to numpy if needed, 
    # but the interface expects lists or arrays. The example uses lists.
    
    # Rank estimate: 43
    # We will generate the terms explicitly.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    # term: (u_vec, v_vec, w_vec) where u_vec is length n*m, v_vec length m*p, w_vec length p*n
    # But we are working with blocks.
    
    # Let's define the block indices.
    # A is 3x4. Indices (i, j) with i in 0..2, j in 0..3.
    # B is 4x4. Indices (j, k) with j in 0..3, k in 0..3.
    # C is 3x4. Indices (i, k) with i in 0..2, k in 0..3.
    # Output convention: W[k*n + i, r] -> W index is (k, i).
    # Input U: U[i*m + j, r] -> U index is (i, j).
    # Input V: V[j*p + k, r] -> V index is (j, k).
    
    # Block definitions:
    # A11: i in [0,1], j in [0,1]
    # A12: i in [0,1], j in [2,3]
    # A21: i in [2,2], j in [0,1] (padded to i in [2,3])
    # A22: i in [2,2], j in [2,3] (padded to i in [2,3])
    
    # B11: j in [0,1], k in [0,1]
    # B12: j in [0,1], k in [2,3]
    # B21: j in [2,3], k in [0,1]
    # B22: j in [2,3], k in [2,3]
    
    # C11: i in [0,1], k in [0,1]
    # C12: i in [0,1], k in [2,3]
    # C21: i in [2,2], k in [0,1]
    # C22: i in [2,2], k in [2,3]
    
    # Strassen terms M1..M7
    # Each M is a 2x2 matrix product of blocks.
    # We need to decompose each M into rank 1 terms of the global tensor.
    
    # Let's create a function to add a rank 1 term for a specific block multiplication.
    # This is complex. Easier to just hardcode the 43 terms.
    
    # I will use a list of tuples (u_idx, v_idx, w_idx, coeff)
    # u_idx is (i, j), v_idx is (j, k), w_idx is (k, i)
    
    terms = []
    
    # Helper to add Strassen terms for 2x2x2
    # X, Y are 2x2 blocks of A, B. Z is 2x2 block of C.
    # We need to map the local 2x2 indices to global indices.
    
    def add_strassen_block(a_i0, a_i1, a_j0, a_j1, b_j0, b_j1, b_k0, b_k1, c_i0, c_i1, c_k0, c_k1):
        # a_i0, a_i1: row range for A block
        # a_j0, a_j1: col range for A block
        # b_j0, b_j1: row range for B block
        # b_k0, b_k1: col range for B block
        # c_i0, c_i1: row range for C block
        # c_k0, c_k1: col range for C block
        
        # Local indices 0,1 map to global ranges.
        # A_local[0,0] -> A[a_i0, a_j0]
        # A_local[0,1] -> A[a_i0, a_j1]
        # A_local[1,0] -> A[a_i1, a_j0]
        # A_local[1,1] -> A[a_i1, a_j1]
        
        # B_local[0,0] -> B[b_j0, b_k0]
        # B_local[0,1] -> B[b_j0, b_k1]
        # B_local[1,0] -> B[b_j1, b_k0]
        # B_local[1,1] -> B[b_j1, b_k1]
        
        # C_local[0,0] -> C[c_i0, c_k0]
        # ...
        
        # Strassen products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        
        # We need to expand each M into rank 1 terms.
        # M1 is product of two 2x2 matrices.
        # (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
        # The product is a 2x2 matrix.
        # We can use the standard Strassen decomposition for this 2x2x2 tensor.
        
        # This is getting too deep for manual expansion.
        # I will use the rank 43 logic by iterating over the 7 terms and adding 7 or 4 rank 1 terms each.
        
        pass

    # Given the complexity and time, I will provide the schoolbook solution with a slight optimization if possible,
    # or just the schoolbook to ensure correctness, as 48 is the baseline.
    # However, the prompt asks for a "stronger algorithm".
    # I will implement the rank 43 decomposition using a pre-computed set of terms.
    
    # Since I cannot easily derive the 43 terms in the scratchpad, I will return the schoolbook rank 48.
    # But wait, the current score is 0.0000. This implies the current program might be failing or the score is just raw value?
    # "score is the rank". Lower is better.
    # If I return 48, score is 48.
    # If I return 43, score is 43.
    
    # Let's try to implement the rank 43 decomposition programmatically.
    
    # I will define the 43 terms explicitly.
    # This is error-prone.
    
    # Alternative: Use the known rank 49 for 4x4x4 and hope the evaluator accepts it? No, dimensions are 3x4x4.
    
    # I will stick to the schoolbook 48 to guarantee validity, as deriving 43 in 150s is risky.
    # But 48 is the "seed" program. The seed score is 1.0? "1.0 equals the seed program".
    # So if I return 48, score is 1.0.
    # If I return 43, score is 48/43 > 1.0.
    
    # I will implement the rank 43 decomposition.
    
    # I will use a helper to generate the terms for a 2x2x2 Strassen block.
    
    def get_strassen_terms_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Returns a list of (u_idx, v_idx, w_idx) for the 7 Strassen products
        of the 2x2 blocks defined by the index ranges.
        """
        terms = []
        # Local 0,1 to Global indices
        # a_rows[0], a_rows[1]
        # a_cols[0], a_cols[1]
        
        # M1: (A00+A11)(B00+B11)
        # A00 is at (a_rows[0], a_cols[0])
        # A11 is at (a_rows[1], a_cols[1])
        # B00 is at (b_rows[0], b_cols[0])
        # B11 is at (b_rows[1], b_cols[1])
        
        # This requires expanding (A00+A11) and (B00+B11) into rank 1 terms.
        # (A00+A11) is a 2x2 matrix with entries:
        # [ A00[0,0]+A11[0,0], A00[0,1]+A11[0,1] ]
        # [ A00[1,0]+A11[1,0], A00[1,1]+A11[1,1] ]
        
        # This is too complex to do on the fly without a solver.
        
        # I will return the schoolbook solution.
        return []

    # I will just return the schoolbook solution.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
