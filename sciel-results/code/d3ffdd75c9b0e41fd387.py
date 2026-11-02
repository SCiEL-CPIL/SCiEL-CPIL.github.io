import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 112 solution:
    # Decompose 4x4x7 into 4x4x4 (Strassen, rank 28) + 4x4x3 (Schoolbook, rank 48) + 4x4x0? No.
    # 4x4x4 part: Strassen rank 28.
    # 4x4x3 part: Schoolbook rank 48.
    # Total rank = 28 + 48 = 76.
    
    R = 76
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for 4x4x4 (cols 0-3 of B) ---
    # A is 4x4, B is 4x4 (first 4 cols), C is 4x4 (first 4 rows of output? No, output is 4x7, so C is 4x4 for first 4 cols)
    # Indices: i in 0..3, j in 0..3, k in 0..3
    
    # Strassen for 2x2 blocks of 2x2 matrices.
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Each P is a 2x2 matrix product.
    # C11 = P1 - P4 + P5 - P6
    # C12 = P3 + P6
    # C21 = P2 + P7
    # C22 = P1 - P3 + P4 + P7
    
    # We need to expand each P into scalar terms.
    # Since P = X * Y where X and Y are 2x2 matrices,
    # P[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # P[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # P[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # P[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    
    # We will define the linear forms for U and V explicitly for each of the 7 products,
    # and for each of the 4 scalar outputs of the 2x2 product.
    # Total 7 * 4 = 28 terms.
    
    # Helper to add a term
    def add_term(i, j, k, coeff):
        nonlocal col
        U[i * m + j][col] += coeff
        V[j * p + k][col] += 1
        W[k * n + i][col] += 1
        col += 1
    
    # Define the Strassen products in terms of indices
    # A indices: i (row), j (col)
    # B indices: j (row), k (col)
    
    # P1 = (A11+A22)(B11+B22)
    # A11: i in 0,1; j in 0,1
    # A22: i in 2,3; j in 2,3
    # B11: j in 0,1; k in 0,1
    # B22: j in 2,3; k in 2,3
    
    # P1[0,0] = (A11+A22)[0,0] * (B11+B22)[0,0] + (A11+A22)[0,1] * (B11+B22)[1,0]
    # (A11+A22)[0,0] = A[0,0] + A[2,2]
    # (A11+A22)[0,1] = A[0,1] + A[2,3]
    # (B11+B22)[0,0] = B[0,0] + B[2,2]
    # (B11+B22)[1,0] = B[1,0] + B[3,2]
    
    # This expansion is tedious. Let's use a more systematic approach.
    # We can generate the 28 terms by iterating over the Strassen formulas.
    
    # Let's define the 7 products P1..P7.
    # For each P, we have 4 scalar terms corresponding to the 4 entries of the 2x2 product.
    # We need to map these to the final C entries.
    
    # Instead of expanding manually, let's use the known rank-28 decomposition for 4x4x4.
    # Or, simpler: use the Kronecker product of 2x2x2 Strassen and 2x2x2 Schoolbook?
    # Rank of 4x4x4 is 28.
    
    # Let's just implement the 28 terms explicitly.
    # This is error-prone. Let's use a safer approach:
    # Use the fact that 4x4x4 can be decomposed into 7 products of 2x2x2.
    # Each 2x2x2 product has rank 7.
    # Total rank 7 * 4 = 28.
    
    # Let's define the 28 terms by looping over the 7 Strassen products and 4 scalar outputs.
    
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # We will create a list of tuples (i, j, k, coeff) for each term.
    
    # To avoid manual expansion errors, let's use the schoolbook for the 4x4x4 part as well?
    # No, that would be rank 64. Total 64 + 48 = 112.
    # 112 is better than 112? No, 112 is the same as schoolbook for 4x4x7 (16*7=112).
    # Wait, schoolbook for 4x4x7 is 4*4*7 = 112.
    # So if I use Strassen for 4x4x4 (rank 28) and Schoolbook for 4x4x3 (rank 48), total is 76.
    # 76 < 112. So it's an improvement.
    
    # Let's implement the 28 terms for 4x4x4 Strassen.
    
    # I will hardcode the 28 terms based on the standard Strassen algorithm for 4x4 matrices.
    
    # The 28 terms are:
    # For each of the 7 products, we have 4 terms.
    
    # Let's define a helper to add a term to the current column.
    
    # P1 = (A11+A22)(B11+B22)
    # Terms for P1:
    # 1. (A[0,0]+A[2,2]) * (B[0,0]+B[2,2]) -> contributes to C[0,0]
    # 2. (A[0,1]+A[2,3]) * (B[1,0]+B[3,2]) -> contributes to C[0,0]
    # 3. (A[0,0]+A[2,2]) * (B[0,1]+B[2,3]) -> contributes to C[0,1]
    # 4. (A[0,1]+A[2,3]) * (B[1,1]+B[3,3]) -> contributes to C[0,1]
    # 5. (A[1,0]+A[3,2]) * (B[0,0]+B[2,2]) -> contributes to C[1,0]
    # 6. (A[1,1]+A[3,3]) * (B[1,0]+B[3,2]) -> contributes to C[1,0]
    # 7. (A[1,0]+A[3,2]) * (B[0,1]+B[2,3]) -> contributes to C[1,1]
    # 8. (A[1,1]+A[3,3]) * (B[1,1]+B[3,3]) -> contributes to C[1,1]
    
    # This is getting too complex to hardcode correctly without verification.
    # Let's fall back to a simpler, verified decomposition.
    
    # Let's use the rank 112 schoolbook solution, which is guaranteed correct.
    # Or, let's try to implement the Strassen part using a library or a known correct implementation.
    
    # Since I cannot verify the Strassen expansion easily, I will return the schoolbook solution.
    
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
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
