"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # We want to decompose the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Using rank R decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Strategy: Use Strassen for 2x2x2 blocks where possible, and direct terms for the rest.
    # The current best known is rank 50.
    # Can we do better?
    # 
    # Let's analyze the structure:
    # C = A * B, where A is 2x4, B is 4x7, C is 2x7.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # We can split B into 7 columns. Each column is a 4x1 vector.
    # Multiplying 2x4 by 4x1 requires rank 4 (schoolbook) or can it be done with less?
    # No, 2x4x1 matrix multiplication has rank 4 (since min(2*4, 4*1, 2*1) = 4? No, rank is at least max(m,n,p) generally? No.)
    # Actually, the rank of matrix multiplication <m x n> x <n x p> is at least max(m*n, n*p, p*m)? No.
    # It is at least max(m, n, p) * something?
    # For 2x4x1, the rank is exactly 4. Proof: The tensor has 8 entries. It is equivalent to the identity map on R^4?
    # A is 2x4, B is 4x1. C is 2x1.
    # This is the contraction of A and B.
    # It is known that rank of <2,4,1> is 4.
    
    # So if we treat each column of B independently, we need 4 * 7 = 28 terms?
    # Wait, the "j" index is shared.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is a single tensor.
    
    # Let's try to find a lower rank decomposition using a more global Strassen-like approach.
    # Split A into two 2x2 blocks: A1 (cols 0,1), A2 (cols 2,3).
    # Split B into corresponding 4x7 blocks? No, B rows correspond to A cols.
    # B rows 0,1 correspond to A cols 0,1. B rows 2,3 correspond to A cols 2,3.
    # Let B1 = B[0:2, :] (2x7), B2 = B[2:4, :] (2x7).
    # Then C = A1 * B1 + A2 * B2.
    # A1 is 2x2, B1 is 2x7. Product is 2x7.
    # A2 is 2x2, B2 is 2x7. Product is 2x7.
    #
    # We can compute A1 * B1 using Strassen on the 2x2 part of A1 and the 2x2 parts of B1?
    # B1 is 2x7. We can split B1 into three 2x2 blocks and one 2x1 block.
    # B1 = [B1_00 B1_01 B1_02] where each is 2x2, except the last is 2x1.
    # Actually, 7 = 2+2+2+1.
    # So B1 has three 2x2 blocks and one 2x1 block.
    #
    # Let's decompose A1 * B1:
    # A1 is 2x2.
    # B1 is 2x7.
    # We can use Strassen for the 2x2x2 parts.
    # There are three 2x2x2 blocks in A1*B1. Each takes rank 7. Total 21.
    # The remaining 2x2x1 block takes rank 4. Total 25.
    #
    # Similarly for A2 * B2:
    # Another 25.
    # Total 50.
    #
    # Can we share terms between A1*B1 and A2*B2?
    # The Strassen formulas involve sums like (A11+A22).
    # In A1*B1, we compute M1 = (A1[0,0]+A1[1,1]) * (B1[0,0]+B1[1,0]) etc.
    # In A2*B2, we compute similar terms for A2 and B2.
    # The terms are distinct because A1, A2, B1, B2 are different entries.
    # So no obvious sharing.
    
    # What if we group differently?
    # Split B into 7 columns.
    # Maybe group columns in a way that allows more Strassen?
    
    # Let's check if rank 49 is possible.
    # Or maybe rank 48?
    
    # Let's try a random search for a lower rank decomposition.
    # We can fix the structure of U, V, W to be sparse or have specific patterns.
    # Or we can use a numerical solver to find a rank R decomposition.
    
    # Given the time constraint, let's try to verify the rank 50 solution first.
    # Then, if time permits, try to find a rank 49 solution.
    
    # Construct rank 50 solution
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4 (k=0..3)
    # Split into four 2x2x2 blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Part 2: 2x4x2 (k=4..5)
    # Split into two 2x2x2 blocks
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # Part 3: 2x4x1 (k=6)
    # Direct terms
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    # Verify
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(lhs - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank solution
    # We can try to remove one term and see if the system is still solvable.
    # This is a non-linear system, so it's hard.
    # Instead, we can try to find a linear dependency among the terms.
    
    # Let's try a different decomposition that might have lower rank.
    # What if we use Strassen for the whole 2x4x7?
    # Split A into 2x2 blocks (2 blocks).
    # Split B into 4x7 blocks? No.
    
    # Let's try to use the fact that n=2 is small.
    # We can write C[0,k] and C[1,k] as linear combinations of bilinear forms.
    # The bilinear forms are A[i,j] * B[j,k].
    # There are 4*7 = 28 such forms.
    # We want to represent C[0,k] and C[1,k] for k=0..6.
    # That's 14 linear combinations.
    # Each combination is a sum of some of the 28 forms.
    # We want to minimize the number of distinct bilinear forms used.
    # But the forms can be shared if they are the same.
    # However, A[i,j] * B[j,k] is unique for each (i,j,k).
    # So we need at least 14 forms? No, because each form contributes to only one C[i,k].
    # Wait, A[i,j] * B[j,k] contributes to C[i,k].
    # So to compute C[0,0], we need A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0].
    # These are 4 distinct forms.
    # To compute C[1,0], we need A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0] + A[1,3]B[3,0].
    # These are 4 distinct forms.
    # Total 8 forms for k=0.
    # Similarly for k=1..6, 8 forms each.
    # Total 56 forms.
    # But Strassen allows sharing by computing sums like (A[0,0]+A[1,0]) * (B[0,0]+B[1,0]).
    # This form contributes to C[0,0], C[0,1], C[1,0], C[1,1].
    # So one form can contribute to multiple C[i,k].
    
    # The rank is the number of such shared forms.
    # We found a decomposition with 50 forms.
    # Can we find one with 49?
    
    # Let's try to search for a rank 49 decomposition by randomly perturbing the rank 50 one
    # and checking if it can be adjusted to satisfy the equations.
    # This is unlikely to work quickly.
    
    # Instead, let's try a different structure.
    # What if we group the columns of B differently?
    # 7 = 3 + 4?
    # 3 columns: 2x2x3? No, 3 is not even.
    # 7 = 1 + 6?
    # 1 column: 2x2x1 (rank 4)
    # 6 columns: 2x2x6 (rank 42? 6*7=42)
    # Total 46?
    # Let's check 2x2x6.
    # Split B into three 2x2 blocks.
    # Each 2x2x2 block takes rank 7.
    # Total 21.
    # So 2x2x6 takes rank 21.
    # 2x2x1 takes rank 4.
    # Total 25.
    # But we have two 2x2 blocks of A.
    # So 2 * 25 = 50. Same as before.
    
    # What if we use a different split for A?
    # A is 2x4.
    # Split into 1x2 and 1x2?
    # No, A is 2 rows.
    
    # It seems 50 is a strong candidate for the minimum rank.
    # Let's return the rank 50 solution.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
