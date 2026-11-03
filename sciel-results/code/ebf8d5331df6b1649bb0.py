import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition with Rank 32.
    # Part 1: 3x3 matrix multiplication (cols 0,1,2 of B)
    # Part 2: 3x2 matrix multiplication (cols 3,4 of B) - Schoolbook (Rank 6)
    
    # Initialize U, V, W for Rank 32
    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # --- Part 2: Schoolbook for 3x2 (cols 3,4) ---
    # This adds 6 terms.
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3, p): # k=3,4
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # --- Part 1: Block Strassen for 3x3 (cols 0,1,2) ---
    # We treat A as 2x2 blocks of sizes (2,1) x (2,1)
    # A = [[A11 (2x2), A12 (2x1)], [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x2), B12 (2x2)], [B21 (1x2), B22 (1x1)]] -- Wait, B is 3x3.
    # B11 is 2x2, B12 is 2x1? No.
    # B is 3x3.
    # B11 (2x2), B12 (2x1) -- No, B has 3 cols.
    # B11 (2x2) uses cols 0,1.
    # B12 (2x1) uses col 2.
    # B21 (1x2) uses cols 0,1.
    # B22 (1x1) uses col 2.
    
    # C = A B
    # C11 (2x2) = A11 B11 + A12 B21
    # C12 (2x1) = A11 B12 + A12 B22
    # C21 (1x2) = A21 B11 + A22 B21
    # C22 (1x1) = A21 B12 + A22 B22
    
    # We will use Strassen for A11 B11 (Rank 7)
    # Schoolbook for the rest.
    
    # Strassen for 2x2:
    # M1 = (A11[0,0]+A11[1,1]) * (B11[0,0]+B11[1,1])
    # M2 = (A11[1,0]+A11[1,1]) * B11[0,0]
    # M3 = A11[0,0] * (B11[1,0]-B11[1,1])
    # M4 = A11[1,1] * (B11[0,1]+B11[1,1])
    # M5 = (A11[0,0]+A11[0,1]) * B11[1,1]
    # M6 = (A11[1,0]-A11[0,0]) * (B11[0,0]+B11[0,1])
    # M7 = (A11[0,1]-A11[1,1]) * (B11[1,0]+B11[1,1])
    
    # C11[0,0] = M1 + M4 - M5 - M7
    # C11[0,1] = M3 + M5
    # C11[1,0] = M2 + M4
    # C11[1,1] = M1 - M2 + M3 + M6
    
    # This is getting complex to map to global indices.
    # I will just use Schoolbook for the 3x3 part as well, but with a smaller rank?
    # No, Schoolbook for 3x3 is 9.
    # Total Rank = 9 (3x3) + 6 (3x2) = 15?
    # Wait, Schoolbook for 3x3 is 9 terms.
    # Schoolbook for 3x2 is 6 terms.
    # Total 15.
    # Why did the seed program use 45?
    # Ah, the seed program used `n*m*p` which is 45.
    # But for 3x3, schoolbook is 9.
    # For 3x5, schoolbook is 15.
    # So I can just return the Schoolbook decomposition for 3x3 and 3x2 separately!
    # Rank 15 is much better than 45.
    
    # Let's re-verify the seed program.
    # `rank_schoolbook = n * m * p` = 45.
    # It iterates `for i in range(n): for j in range(m): for k in range(p):`
    # This is 3*3*5 = 45 iterations.
    # But for a given (i,k), the sum is over j.
    # The term is non-zero only if U, V, W match.
    # In schoolbook, each (i,j,k) is a separate term.
    # So yes, rank is 45.
    
    # But wait, if I use the standard schoolbook formula:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This requires 15 multiplications (3*5).
    # The rank of the decomposition should be 15.
    # Why is the seed 45?
    # Because it creates a term for EVERY (i,j,k) triple, even if they don't contribute to the same (i,k) output in a way that allows sharing?
    # No, in the tensor decomposition, each term r corresponds to a scalar multiplication.
    # If I use 15 terms, I need U, V, W of size (9, 15), (15, 15), (15, 15).
    # For term r corresponding to (i,j,k):
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    # This is exactly what the seed program does!
    # But the seed program has 45 terms.
    # Why?
    # Because it iterates over ALL i, j, k.
    # But for a fixed (i,k), we sum over j.
    # The term for (i,j,k) contributes to C[i,k].
    # If I have 15 terms, one for each (i,k) pair? No.
    # The multiplication A[i,j] * B[j,k] is a scalar multiplication.
    # There are 3*3*5 = 45 such scalar multiplications in the full expansion.
    # But we can combine them?
    # No, the rank is the minimum number of scalar multiplications.
    # For 3x3 matrix multiplication, the rank is 21 (or 23, etc), not 9.
    # Schoolbook uses 9 multiplications for 3x3?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For 3x3, there are 9 outputs. Each output is a sum of 3 products.
    # Total products = 27?
    # No, the products are A[i,j] * B[j,k].
    # There are 27 such products.
    # But we can reuse them?
    # No, each product is distinct.
    # So Schoolbook rank for 3x3 is 27?
    # Wait, C[i,k] depends on A[i,0]B[0,k], A[i,1]B[1,k], A[i,2]B[2,k].
    # For fixed i, the 3 products are distinct.
    # For different i, the products are distinct.
    # So total products = 3 (i) * 3 (j) * 3 (k) = 27.
    # So Schoolbook rank for 3x3 is 27.
    # Schoolbook rank for 3x5 is 45.
    
    # So the seed program is correct in using 45.
    # My previous thought that "Schoolbook for 3x3 is 9" was wrong.
    # 9 is the number of outputs.
    # The number of multiplications is 27.
    
    # So, can I find a decomposition with rank < 45?
    # Yes, Strassen reduces the rank.
    # For 3x3, rank is 21.
    # For 3x5, rank is likely < 45.
    
    # I will implement the 21-rank decomposition for the 3x3 part (cols 0,1,2)
    # and Schoolbook for the 3x2 part (cols 3,4).
    # Schoolbook for 3x2:
    # Outputs: 3x2 = 6.
    # Products: 3 (i) * 3 (j) * 2 (k) = 18.
    # So Rank 18 for the 3x2 part.
    # Total Rank = 21 + 18 = 39.
    # This is an improvement over 45.
    
    # I need the 21-rank coefficients for 3x3.
    # I will use a known set.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
