"""Program implementing Strassen-based rank minimization for 4x4x6 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # Returns lists: a_coefs, b_coefs, c_coefs for each of the 7 terms
    # Term t: C[ca, cb] += c_coefs[t][ca][cb] * (sum_a a_coefs[t][a][b]*A[a][b]) * (sum_c b_coefs[t][c][d]*B[c][d])
    
    # Indexing: A is 2x2 (rows 0,1; cols 0,1), B is 2x2, C is 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We'll represent each term by its A coefficients (2x2), B coefficients (2x2), and C contribution (2x2)
    
    def strassen_2x2():
        """
        Returns a list of 7 dicts, each with keys 'A' (2x2 list), 'B' (2x2 list), 'C' (2x2 list)
        where the bilinear form is sum(A[i][j]*B[i][j]) contributing to C via the C matrix.
        Actually, it's sum_{i,j} A[i][j] * B[i][j] where A and B are linear combinations.
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # A part: A00*1 + A11*1 => coeff [[1,0],[0,1]] for A00,A01,A10,A11? No.
        # A is 2x2. The linear form is L_A = A[0][0] + A[1][1]
        # So A_coeffs = [[1,0],[0,1]]
        # L_B = B[0][0] + B[1][1]
        # B_coeffs = [[1,0],[0,1]]
        # This product contributes to C[0][0] and C[1][1]
        terms.append({
            'A': [[1, 0], [0, 1]],
            'B': [[1, 0], [0, 1]],
            'C': [[1, 0], [0, 1]]  # Wait, C00 gets +M1, C11 gets +M1. So C contrib is [[1,0],[0,1]]
        })
        
        # M2 = (A10+A11)B00
        # L_A = A[1][0] + A[1][1] => [[0,0],[1,1]]
        # L_B = B[0][0] => [[1,0],[0,0]]
        # C10 += M2, C11 -= M2
        terms.append({
            'A': [[0, 0], [1, 1]],
            'B': [[1, 0], [0, 0]],
            'C': [[0, 0], [1, -1]]  # C10 gets +1, C11 gets -1
        })
        
        # M3 = A00(B01-B11)
        # L_A = A[0][0] => [[1,0],[0,0]]
        # L_B = B[0][1] - B[1][1] => [[0,1],[0,-1]]
        # C01 += M3, C11 += M3
        terms.append({
            'A': [[1, 0], [0, 0]],
            'B': [[0, 1], [0, -1]],
            'C': [[0, 1], [0, 1]]  # C01 gets +1, C11 gets +1
        })
        
        # M4 = A11(B10-B00)
        # L_A = A[1][1] => [[0,0],[0,1]]
        # L_B = B[1][0] - B[0][0] => [[-1,0],[1,0]]
        # C00 += M4, C10 += M4
        terms.append({
            'A': [[0, 0], [0, 1]],
            'B': [[-1, 0], [1, 0]],
            'C': [[1, 0], [1, 0]]  # C00 gets +1, C10 gets +1
        })
        
        # M5 = (A00+A01)B11
        # L_A = A[0][0] + A[0][1] => [[1,1],[0,0]]
        # L_B = B[1][1] => [[0,0],[0,1]]
        # C00 -= M5, C01 += M5
        terms.append({
            'A': [[1, 1], [0, 0]],
            'B': [[0, 0], [0, 1]],
            'C': [[-1, 1], [0, 0]]  # C00 gets -1, C01 gets +1
        })
        
        # M6 = (A10-A00)(B00+B01)
        # L_A = A[1][0] - A[0][0] => [[-1,0],[1,0]]
        # L_B = B[0][0] + B[0][1] => [[1,1],[0,0]]
        # C11 += M6
        terms.append({
            'A': [[-1, 0], [1, 0]],
            'B': [[1, 1], [0, 0]],
            'C': [[0, 0], [0, 1]]  # C11 gets +1
        })
        
        # M7 = (A01-A11)(B10+B11)
        # L_A = A[0][1] - A[1][1] => [[0,1],[0,-1]]
        # L_B = B[1][0] + B[1][1] => [[0,0],[1,1]]
        # C00 += M7
        terms.append({
            'A': [[0, 1], [0, -1]],
            'B': [[0, 0], [1, 1]],
            'C': [[1, 0], [0, 0]]  # C00 gets +1
        })
        
        return terms

    strassen_terms = strassen_2x2()
    
    # Build the full decomposition
    # The multiplication tensor T[i,j,k] where C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to decompose this into rank-1 tensors.
    # Each Strassen term for a 2x2x2 block gives a rank-1 decomposition for that block.
    # But we need to handle the 4x4x6 case.
    
    # Strategy:
    # Split the 4x4x6 multiplication into:
    # 1. 4x4x4 part (first 4 columns of B)
    # 2. 4x4x2 part (last 2 columns of B)
    
    # For 4x4x4:
    # Split A into 2x2 blocks: A00, A01, A10, A11
    # Split B into 2x2 blocks: B00, B01, B10, B11 (for the first 4 columns)
    # Split C into 2x2 blocks: C00, C01, C10, C11
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each term is a sum of two 2x2x2 products.
    # We can use Strassen for each 2x2x2 product.
    # So for C00, we have A00*B00 (7 terms) + A01*B10 (7 terms) = 14 terms.
    # Similarly for C01, C10, C11.
    # Total for 4x4x4: 4 * 14 = 56 terms? Wait, no.
    
    # Actually, the standard way is to use Strassen on the whole 4x4x4.
    # But let's think of it as 4 independent 2x2x2 multiplications.
    # Each 2x2x2 has rank 7. So 4 * 7 = 28 terms for 4x4x4?
    # No, that's not right either. The 4x4x4 multiplication has rank at most 49.
    
    # Let's reconsider. The 4x4x4 multiplication can be seen as:
    # C = A * B where A is 4x4, B is 4x4, C is 4x4.
    # Using block decomposition with 2x2 blocks:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each of these is a sum of two 2x2x2 products.
    # We can apply Strassen to each 2x2x2 product.
    # For example, for A00*B00, we have 7 bilinear forms.
    # So for C00, we have 7 (for A00*B00) + 7 (for A01*B10) = 14 bilinear forms.
    # Similarly for C01, C10, C11.
    # Total: 4 * 14 = 56 bilinear forms.
    
    # But wait, can we do better? The rank of 4x4x4 is known to be at most 49.
    # The above gives 56. Maybe we can share some terms?
    # For now, let's use the 56-term decomposition and see if we can improve.
    
    # Actually, let's just implement the direct application of Strassen to each 2x2x2 subproduct.
    
    # For 4x4x2:
    # B is 4x2. We can split it into B_top (2x2) and B_bottom (2x2).
    # C is 4x2, split into C_top (2x2) and C_bottom (2x2).
    # C_top = A00*B_top + A01*B_bottom
    # C_bottom = A10*B_top + A11*B_bottom
    # Each is a sum of two 2x2x2 products.
    # So 2 * 14 = 28 terms.
    
    # Total rank: 56 + 28 = 84.
    # Schoolbook is 4*4*6 = 96.
    # So we save 12 ranks.
    
    # Let's implement this.
    
    # First, let's create a helper to add a Strassen term for a 2x2x2 block.
    # The block is defined by:
    # A block: rows a0..a0+1, cols b0..b0+1
    # B block: rows c0..c0+1, cols d0..d0+1
    # C block: rows e0..e0+1, cols f0..f0+1
    # But in our case, the indices are fixed by the block decomposition.
    
    # Let's define the global indices.
    # A is 4x4: rows 0..3, cols 0..3
    # B is 4x6: rows 0..3, cols 0..5
    # C is 4x6: rows 0..3, cols 0..5
    
    # For the 4x4x4 part (cols 0..3 of B and C):
    # A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B blocks (for cols 0-3):
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C blocks:
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # For the 4x4x2 part (cols 4-5 of B and C):
    # B_top: rows 0-1, cols 4-5
    # B_bottom: rows 2-3, cols 4-5
    # C_top: rows 0-1, cols 4-5
    # C_bottom: rows 2-3, cols 4-5
    
    # Now, let's create the U, V, W matrices.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    # We'll start with rank = 0 and add terms one by one.
    
    def add_term(U, V, W, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, term):
        """
        Add a Strassen term to U, V, W.
        a_rows, a_cols: row and col indices for the 2x2 A block
        b_rows, b_cols: row and col indices for the 2x2 B block
        c_rows, c_cols: row and col indices for the 2x2 C block
        term: dict with 'A', 'B', 'C' keys, each a 2x2 list
        """
        # Create a new column in U, V, W
        col = len(U[0])
        
        # Extend U, V, W
        for i in range(len(U)):
            U[i].append(0)
        for i in range(len(V)):
            V[i].append(0)
        for i in range(len(W)):
            W[i].append(0)
        
        # Fill U for this term
        for i in range(2):
            for j in range(2):
                global_i = a_rows + i
                global_j = a_cols + j
                U[global_i * m + global_j][col] = term['A'][i][j]
        
        # Fill V for this term
        for i in range(2):
            for j in range(2):
                global_i = b_rows + i
                global_j = b_cols + j
                V[global_i * p + global_j][col] = term['B'][i][j]
        
        # Fill W for this term
        for i in range(2):
            for j in range(2):
                global_i = c_cols + j  # col index in C
                global_j = c_rows + i  # row index in C
                # W is indexed as [k*n + i, rank] where k is col, i is row
                # So W[global_k * n + global_i]
                W[global_i * n + global_j][col] = term['C'][i][j]
        
        return col
    
    # Initialize U, V, W
    U = [[0] * 0 for _ in range(n * m)]
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    # 4x4x4 part
    # C00 = A00*B00 + A01*B10
    for t in strassen_terms:
        add_term(U, V, W, 0, 0, 0, 0, 0, 0, t)  # A00, B00, C00
        add_term(U, V, W, 0, 2, 2, 0, 0, 0, t)  # A01, B10, C00
    
    # C01 = A00*B01 + A01*B11
    for t in strassen_terms:
        add_term(U, V, W, 0, 0, 0, 2, 0, 2, t)  # A00, B01, C01
        add_term(U, V, W, 0, 2, 2, 2, 0, 2, t)  # A01, B11, C01
    
    # C10 = A10*B00 + A11*B10
    for t in strassen_terms:
        add_term(U, V, W, 2, 0, 0, 0, 2, 0, t)  # A10, B00, C10
        add_term(U, V, W, 2, 2, 2, 0, 2, 0, t)  # A11, B10, C10
    
    # C11 = A10*B01 + A11*B11
    for t in strassen_terms:
        add_term(U, V, W, 2, 0, 0, 2, 2, 2, t)  # A10, B01, C11
        add_term(U, V, W, 2, 2, 2, 2, 2, 2, t)  # A11, B11, C11
    
    # 4x4x2 part
    # C_top = A00*B_top + A01*B_bottom
    for t in strassen_terms:
        add_term(U, V, W, 0, 0, 0, 4, 0, 4, t)  # A00, B_top, C_top
        add_term(U, V, W, 0, 2, 2, 4, 0, 4, t)  # A01, B_bottom, C_top
    
    # C_bottom = A10*B_top + A11*B_bottom
    for t in strassen_terms:
        add_term(U, V, W, 2, 0, 0, 4, 2, 4, t)  # A10, B_top, C_bottom
        add_term(U, V, W, 2, 2, 2, 4, 2, 4, t)  # A11, B_bottom, C_bottom
    
    rank = len(U[0])
    
    return {"U": U, "V": V, "W": W}
