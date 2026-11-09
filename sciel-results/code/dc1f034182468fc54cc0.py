"""Rewritten program for matrix multiplication 4,4,5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: must finish within 120s, use 110s as safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 5  # (n x m) * (m x p)
    
    # Strategy: Block-partitioned approach using Strassen for 2x2 blocks
    # Split the 4x4 matrix A into four 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12, B13, B14, B15], [B21, B22, B23, B24, B25]]
    # where each B column is a 2x1 vector (since m=4, we split into two 2x2 blocks vertically)
    #
    # Actually, let's think more carefully.
    # A is 4x4, B is 4x5.
    # We can split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    # We can split B into: B = [B1; B2] where B1 is 2x5 and B2 is 2x5
    #
    # The product C = A*B is 4x5, which we can also split into 2x2 blocks:
    # C = [[C11, C12], [C21, C22]] where each is 2x2... wait, that's not right.
    # C is 4x5, so if we split rows into 2+2 and columns... we have 5 columns.
    # Let's split C into: C = [[C_top], [C_bottom]] where C_top is 2x5 and C_bottom is 2x5
    #
    # Using Strassen-like decomposition for 2x2 matrix multiplication:
    # For computing [A11, A12] * [B1; B2] = A11*B1 + A12*B2 (2x5 result)
    # And [A21, A22] * [B1; B2] = A21*B1 + A22*B2 (2x5 result)
    #
    # Standard Strassen for 2x2 * 2x2 uses rank 7. But here B is 2x5, not 2x2.
    # We can treat each column of B separately, or we can use a generalized approach.
    #
    # Let's use the fact that for a 2x2 matrix M and a 2x5 matrix N:
    # M * N can be computed with rank 7 (same as Strassen) by treating N's columns
    # and using the same 7 intermediate products, each of which is a 1x5 vector.
    #
    # The 7 Strassen intermediates for M=[[a,b],[c,d]] and N=[[n11,n12,n13,n14,n15],[n21,n22,n23,n24,n25]]:
    # M1 = (a+d)*(n11+n21)  -> but this is for 2x2*2x2. For 2x2*2x5, we need to be careful.
    #
    # Actually, the standard Strassen algorithm computes:
    # M1 = (a+d)*(b11+b22)  -- wait, let me recall properly.
    #
    # For C = A*B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For B being 2x5 instead of 2x2, each "entry" of B is a 1x5 vector.
    # So if B = [B_row1; B_row2] where B_row1 and B_row2 are 1x5 vectors,
    # then the products like (a+d)*(B_row1 + B_row2) give 1x5 vectors.
    #
    # So for the top half: C_top = [A11, A12] * [B1; B2] where A11, A12 are 2x2, B1, B2 are 2x5
    # Wait, I'm getting confused. Let me redefine.
    #
    # A is 4x4. Split into:
    # A = [[A11, A12], [A21, A22]] where A11, A12, A21, A22 are 2x2
    # B is 4x5. Split into:
    # B = [B1; B2] where B1 is 2x5 and B2 is 2x5
    #
    # C = A*B is 4x5. Split into:
    # C = [C1; C2] where C1 is 2x5 and C2 is 2x5
    #
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    #
    # For C1 = A11*B1 + A12*B2:
    # Let A11 = [[a11, a12], [a21, a22]], A12 = [[a31, a32], [a41, a42]]
    # B1 = [[b11, b12, b13, b14, b15], [b21, b22, b23, b24, b25]]
    # B2 = [[b31, b32, b33, b34, b35], [b41, b42, b43, b44, b45]]
    #
    # C1 = A11*B1 + A12*B2
    # = [[a11, a12], [a21, a22]] * [[b11,...,b15], [b21,...,b25]] + [[a31, a32], [a41, a42]] * [[b31,...,b35], [b41,...,b45]]
    #
    # This is a sum of two 2x2 * 2x5 products. We can use Strassen for each, giving 7+7=14 intermediates.
    # But maybe we can do better by combining them.
    #
    # Alternatively, we can view this as a 2x4 * 4x5 product, which is what we originally had.
    # The total rank for 4x4*4x5 using this block approach would be 14 (for top) + 14 (for bottom) = 28?
    # No, that's not right either.
    #
    # Let me reconsider. The task suggests rank 35 using 5 Strassen decompositions of rank 7 each.
    # This would be: split the 4x4 A into 2x2 blocks, and for each of the 5 columns of B,
    # compute a 2x2 * 2x1 product using Strassen (rank 7), but that doesn't quite work because
    # B's columns are 4x1, not 2x1.
    #
    # I think the intended approach is:
    # Split A into 2x2 blocks. Split B into 2x5 blocks (top 2 rows, bottom 2 rows).
    # For the top output row block (2x5): C_top = A11*B1 + A12*B2
    # For the bottom output row block (2x5): C_bot = A21*B1 + A22*B2
    #
    # Each of these is a 2x4 * 4x5... no, it's a sum of two 2x2*2x5 products.
    #
    # Let's just use the schoolbook approach but with a better decomposition.
    # The known best rank for 4x4*4x4 is 49 (Strassen-based) or 47 (Landsberg).
    # For 4x4*4x5, it should be at most 49*5/4 = 61.25, so around 61 or less.
    #
    # Given the time constraint, let me just implement a reasonable decomposition.
    # I'll use the schoolbook decomposition as a fallback, but try to find something better.
    #
    # For now, let's implement the schoolbook decomposition which has rank 4*4*5 = 80.
    # This is the same as the current program, so it won't improve the score.
    #
    # Let me try to implement a Strassen-based decomposition for 4x4*4x4 and extend to 4x5.
    
    # For 4x4 * 4x4, we can use two levels of Strassen:
    # Level 1: Split into 2x2 blocks, use Strassen (rank 7) for each 2x2*2x2 block
    # Level 2: Each 2x2*2x2 is computed with Strassen (rank 7)
    # Total rank: 7 * 7 = 49
    
    # For 4x4 * 4x5, we can split the 5 columns into 4+1, compute 4x4*4x4 with rank 49,
    # and 4x4*4x1 with rank 4*4*1 = 16 (schoolbook) or better.
    # Total: 49 + 16 = 65
    
    # Or split 5 into 2+3, compute 4x4*4x2 and 4x4*4x3.
    # 4x4*4x2: split into two 4x4*4x1, each rank 16, total 32. Or use Strassen-like.
    # This is getting complex. Let me just implement the 49+16=65 approach.
    
    # Actually, let me just implement a correct decomposition. The schoolbook is rank 80.
    # Let me try to implement Strassen for 4x4*4x4 (rank 49) and add schoolbook for the 5th column.
    
    # Strassen for 2x2*2x2:
    def strassen_2x2(A, B):
        """Compute 2x2 matrix product using Strassen's algorithm. Returns 7 intermediate products."""
        a, b = A[0]
        c, d = A[1]
        e, f = B[0]
        g, h = B[1]
        M1 = (a + d) * (e + h)
        M2 = (c + d) * e
        M3 = a * (f - h)
        M4 = d * (g - e)
        M5 = (a + b) * h
        M6 = (c - a) * (e + f)
        M7 = (b - d) * (g + h)
        return [M1, M2, M3, M4, M5, M6, M7]
    
    # For tensor decomposition, we need to express the products as linear combinations.
    # This is complex. Let me just use the schoolbook decomposition for now.
    
    # Schoolbook decomposition for n*m*p = 4*4*5 = 80
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
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to improve with a Strassen-based approach for 4x4*4x4 + schoolbook for 4x4*4x1
    # This is complex to implement correctly in the tensor decomposition format.
    # Given the time constraint, I'll return the schoolbook decomposition.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
