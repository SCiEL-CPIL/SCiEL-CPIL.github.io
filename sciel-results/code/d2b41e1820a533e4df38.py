"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = n*m*p = 112 for schoolbook
    # We'll use Strassen for the 4x4 x 4x4 part (top 4 rows of B)
    # and schoolbook for the 4x4 x 4x3 part (bottom 3 rows of B)
    
    # For 4x4 x 4x4 using Strassen at top level:
    # Split A into 2x2 blocks, B into 2x2 blocks
    # 7 Strassen products, each is 2x2 x 2x2 schoolbook = 8 scalar mults
    # So 7*8 = 56 terms for the 4x4 x 4x4 part
    # For 4x4 x 4x3: 4*4*3 = 48 terms
    # Total rank = 56 + 48 = 104
    
    total_rank = 104
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Strassen 2x2 x 2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Strassen combination coefficients for the 7 products
    # Each product is a 2x2 matrix; we need to express each element as sum of scalar mults
    # For a 2x2 schoolbook product X*Y where X is 2x2 and Y is 2x2:
    # (X*Y)[i,k] = sum_j X[i,j]*Y[j,k]
    # Each X[i,j] is a linear combination of A entries, each Y[j,k] is a linear combo of B entries
    # So each (X*Y)[i,k] = sum over (a_idx, b_idx) of coeff_a * coeff_b * A[a_idx]*B[b_idx]
    
    # Let's define the Strassen linear combinations
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    # For each Strassen product r in 0..6, and each output position (i,k) in {0,1}^2:
    # We need to expand X[i,j]*Y[j,k] where X and Y are the 2x2 matrices being multiplied
    
    # Define the linear forms for each Strassen product
    # For product r, X_r and Y_r are 2x2 matrices with entries that are linear combos of A and B entries
    
    # Let's build this systematically
    # A entry indices: (i,j) for i in 0..3, j in 0..3
    # B entry indices: (j,k) for j in 0..3, k in 0..3
    
    # Strassen product definitions (X_r, Y_r):
    # M1: X = [[A00,A00],[A11,A11]]... no, let me be more careful
    
    # Actually, for Strassen 2x2x2x2:
    # M1 = (A00+A11)(B00+B11)
    #   X = [[A00,A00],[A11,A11]]? No.
    #   (A00+A11) is a 2x2 matrix: [[a00+a22, a01+a23],[a10+a32,a11+a33]]
    #   (B00+B11) is a 2x2 matrix: [[b00+b20, b01+b21],[b10+b30,b11+b31]]
    #   So X = [[a00+a22, a01+a23],[a10+a32, a11+a33]]
    #   Y = [[b00+b20, b01+b21],[b10+b30, b11+b31]]
    #   Product X*Y is 2x2, each entry is sum of 2 scalar mults
    
    # This is getting complex. Let me just enumerate all scalar mults for each of the 7 Strassen products.
    
    # For each Strassen product r, define X_r and Y_r as 2x2 matrices of linear forms
    # Then (X_r * Y_r)[i][k] = sum_j X_r[i][j] * Y_r[j][k]
    # Each X_r[i][j] = sum of A entries with coefficients
    # Each Y_r[j][k] = sum of B entries with coefficients
    
    # Let me define helper: for each product, list of (X_row, X_col, X_coeffs, Y_row, Y_col, Y_coeffs)
    # where X_coeffs is dict from (i,j) to coeff, Y_coeffs is dict from (j,k) to coeff
    
    # Actually, let me just hardcode the 56 terms by thinking about it differently.
    
    # Alternative: For the top 4 rows of B (B[:, 0:4]), we can think of it as multiplying
    # A (4x4) by B_top (4x4) using Strassen. The result C_top (4x4) has entries C[i][k] for k in 0..3.
    # Using Strassen, C_top is expressed as sum of 7 2x2 matrix products.
    # Each 2x2 product contributes 4 output entries, each of which is a sum of scalar mults.
    
    # Let me just directly compute the 56 rank-1 terms for the Strassen decomposition of 4x4 x 4x4.
    
    # I'll define the 7 Strassen products and their expansions.
    
    # For M1 = (A00+A11)(B00+B11):
    # X1 = [[a00+a22, a01+a23], [a10+a32, a11+a33]]
    # Y1 = [[b00+b20, b01+b21], [b10+b30, b11+b31]]
    # (X1*Y1)[0][0] = (a00+a22)(b00+b20) + (a01+a23)(b10+b30)
    #   = a00*b00 + a00*b20 + a22*b00 + a22*b20 + a01*b10 + a01*b30 + a23*b10 + a23*b30
    # That's 8 scalar mults for just one entry!
    
    # Wait, that's not right for tensor rank. Each scalar mult a_ij * b_jk is one rank-1 term.
    # So (X1*Y1)[0][0] alone contributes 8 rank-1 terms.
    # The full 2x2 product X1*Y1 has 4 entries, each with 8 terms, but some may overlap.
    # Actually no, they're different output positions so they don't overlap in the tensor.
    
    # Hmm, but that gives 7 * 4 * 8 = 224 terms, which is way more than schoolbook.
    
    # I think I'm misunderstanding how Strassen reduces rank. Strassen reduces the number of 
    # multiplicative operations but when you expand back to scalar level, you might not get 
    # a direct reduction in tensor rank unless you're careful.
    
    # Actually, the key insight is that Strassen works at the level of field extensions or 
    # polynomial rings. For tensor rank over a field, Strassen's 7-product algorithm for 
    # 2x2 matrices gives a rank-7 decomposition of the 2x2x2 multiplication tensor.
    # But that's for 2x2x2, not 4x4x4.
    
    # For 4x4x4, the known upper bound is 49 (from Strassen applied recursively: 7^2 = 49).
    # But constructing the explicit rank-49 decomposition is non-trivial.
    
    # Given the complexity, let me just use the schoolbook decomposition with rank 112.
    # This is the safest approach and matches the seed score.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
