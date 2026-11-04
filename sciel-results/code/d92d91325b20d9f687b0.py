import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to express these in terms of the 4x4 blocks.
    # A is 4x4, split into 2x2 blocks of 2x2 matrices.
    # A11, A12, A21, A22 are 2x2 matrices.
    # B is 4x4 (or 4x2 padded), split into 2x2 blocks of 2x2 (or 2x1) matrices.
    
    # Let's define the linear combinations for A and B.
    # Each term in the schoolbook decomposition of X*Y is x_ab * y_cd * E_ad.
    # We need to map a,b,c,d to the global indices i,j,k.
    
    # For a 2x2 block at position (block_i, block_j) of A, the local indices (a,b) map to global (i,j) as:
    # i = block_i * 2 + a
    # j = block_j * 2 + b
    
    # For a 2x2 block at position (block_j, block_k) of B, the local indices (c,d) map to global (j,k) as:
    # j = block_j * 2 + c
    # k = block_k * 2 + d
    
    # The Strassen products involve sums/differences of these blocks.
    # We will iterate through the 7 Strassen products.
    # For each product, we have linear combinations of A blocks and B blocks.
    # Let L be the linear combination of A blocks, R be the linear combination of B blocks.
    # L is a 2x2 matrix of coefficients for the 4 A blocks.
    # R is a 2x2 matrix of coefficients for the 4 B blocks.
    # Actually, L and R are 2x2 matrices of 2x2 matrices? No.
    # L is a 2x2 matrix where each entry is a linear combination of the 4 A blocks.
    # This is getting complicated.
    
    # Alternative: Just expand the Strassen formula into the 28 terms directly.
    # Each term is of the form (linear_comb_A) * (linear_comb_B) * E_{ik}.
    # The linear_comb_A is a sum of A_{ij} with coefficients.
    # The linear_comb_B is a sum of B_{jk} with coefficients.
    
    # Let's list the 28 terms for M_4,4,4.
    # Term t corresponds to a specific i,j,k and coefficients for U,V,W.
    # Actually, U,V,W entries are 0 or 1 (or -1).
    
    # We can generate the terms by iterating over the 7 Strassen products and their 4 schoolbook components.
    
    strassen_terms = []
    
    # Helper to add a term
    def add_term(i, j, k, coeff_u, coeff_v, coeff_w):
        # i,j,k are global indices for A, B, C
        # coeff_u, coeff_v, coeff_w are the coefficients for U, V, W
        # In our case, they are +/- 1.
        strassen_terms.append((i, j, k, coeff_u, coeff_v, coeff_w))
        
    # Define the 7 Strassen products
    # M1 = (A11+A22)(B11+B22)
    # A11: i in [0,1], j in [0,1]
    # A22: i in [2,3], j in [2,3]
    # B11: j in [0,1], k in [0,1]
    # B22: j in [2,3], k in [2,3]
    
    # We need to decompose the product of sums.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # Each of these is a 2x2 * 2x2 product.
    # Each 2x2 * 2x2 product has 4 schoolbook terms.
    # So M1 contributes 4 * 4 = 16 terms?
    # No, Strassen's algorithm replaces the 8 schoolbook terms with 7 products.
    # But here we are using Strassen to reduce the number of *multiplications*.
    # The "multiplications" are of 2x2 matrices.
    # If we expand each 2x2 multiplication into schoolbook, we get 4 terms per multiplication.
    # So M1 contributes 4 terms.
    # M2 contributes 4 terms.
    # ...
    # M7 contributes 4 terms.
    # Total 28 terms.
    
    # Let's define the blocks.
    # A blocks: A00, A01, A10, A11 (using 0-based block indices)
    # B blocks: B00, B01, B10, B11
    
    # M1 = (A00 + A11)(B00 + B11)
    # This is a product of two 2x2 matrices.
    # Let X = A00 + A11, Y = B00 + B11.
    # X is a 2x2 matrix. Y is a 2x2 matrix.
    # The schoolbook terms for X*Y are:
    # (X00*Y00) E00, (X01*Y10) E01, (X10*Y01) E10, (X11*Y11) E11.
    # X00 = A00_00 + A11_00.
    # This is getting too detailed.
    
    # I will use a simpler approach:
    # Generate the rank-28 decomposition of M_4,4,4 using a known algorithm or by expanding Strassen.
    # Since I don't have the explicit 28 terms, I will use the schoolbook decomposition for M_4,4,4 (rank 64)
    # and see if I can improve it.
    # But 64 is not better than 96.
    
    # Wait, the rank of M_4,4,4 is 64 with schoolbook.
    # My previous calculation of rank 28 was based on Strassen.
    # If I can't easily generate the 28 terms, I might stick to a simpler improvement.
    
    # What if I use Strassen for the 4x4 part and schoolbook for the rest?
    # Rank 49 for 4x4x4.
    # Rank 32 for 4x4x2.
    # Total 81.
    # Score 96/81 = 1.185.
    
    # This is better than 1.0.
    # I will implement the rank-81 decomposition.
    
    # Step 1: Generate Strassen decomposition for 4x4x4.
    # Strassen uses 7 products of 2x2 matrices.
    # Each 2x2 product uses schoolbook (rank 4).
    # Total rank 28? No, Strassen rank is 49.
    # Why?
    # Strassen's algorithm for 2x2 has rank 7.
    # For 4x4, we use 7 Strassen products.
    # Each Strassen product is a 2x2 matrix multiplication.
    # If we use Strassen for these, rank is 7*7=49.
    # If we use schoolbook for these, rank is 7*4=28.
    # Is rank 28 valid?
    # Yes, as argued before.
    # So I should use rank 28 for 4x4x4.
    
    # I will implement the rank-28 decomposition for 4x4x4.
    
    # I will hardcode the 28 terms or generate them.
    # Generating them is safer.
    
    # I will use a helper to generate the terms for a 2x2 Strassen product.
    
    # Let's assume the 28 terms are generated.
    # I will create the U, V, W matrices.
    
    # For now, I will return the schoolbook decomposition to ensure correctness,
    # and then refine it.
    
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
