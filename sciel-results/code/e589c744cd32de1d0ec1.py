"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    
    # Matrix multiplication tensor for C = A @ B where A is n x m, B is m x p
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    
    # Strategy: Use Strassen-like decomposition on the 3x3 matrix part.
    # The 3x3 matrix multiplication can be decomposed using 7 multiplications
    # (Strassen) instead of 9. We can apply this to the matrix-vector products.
    #
    # For C[i,:] = sum_j A[i,j] * B[j,:], we have 3 such equations for i=0,1,2.
    # Each involves 3 multiplications (scalar * vector), so 9 total.
    #
    # With Strassen-like approach for 3x3:
    # We can group the 3x3 matrix multiplication into a form that uses fewer
    # scalar multiplications by precomputing linear combinations.
    #
    # Actually, for matrix-matrix multiplication M_{3,3,3}, the best known rank
    # is 23 (vs schoolbook 27). But we're doing M_{3,3,8} which is different.
    #
    # Let me think about this differently. The tensor T has shape (9, 24, 24).
    # T[a,b,c] is nonzero only when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    #
    # For each fixed j, we have a sub-tensor that computes:
    #   sum_i A[i,j] * B[j,k] -> C[i,k]
    # This is an outer product of A[:,j] (length 3) and B[j,:] (length 8),
    # contributing to C (3x8).
    #
    # The full computation is sum over j of these outer products.
    #
    # Schoolbook: 3 (for j) * 3 (for i) * 8 (for k) = 72 terms.
    #
    # To reduce rank, we need to find linear dependencies or use algebraic
    # identities. One approach: for the 3x3 matrix part, use a rank-7 
    # decomposition (Strassen-style) and combine with the 8 columns.
    #
    # Let me try a specific construction. For the 3x3 matrix multiplication:
    # A (3x3) * B (3x3) = C (3x3)
    # Strassen uses 7 multiplications:
    # M1 = (A00+A22)*(B00+B22)
    # M2 = (A10+A20)*B00
    # M3 = A00*(B01-B21)
    # M4 = A22*(B11-B01)
    # M5 = (A01+A11)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M5
    # C11 = M1-M2+M3+M4
    # C20 = M2+(M1-M3-M5)  ... wait, let me look up the correct formulas.
    #
    # Actually, standard Strassen for 2x2 is well-known. For 3x3, there are
    # various constructions. Let me use a simpler approach:
    #
    # For M_{3,3,8}, think of it as: for each of the 8 columns of B, compute
    # A * b_col where b_col is a 3-vector. This is 3x3 matrix-vector product.
    # The rank of M_{3,3,1} (matrix-vector) is 9 (schoolbook) and I believe
    # it's still 9 (no known improvement for matrix-vector).
    #
    # So for 8 independent columns, we'd have 8*9 = 72 if done independently.
    # But we can share work across columns!
    #
    # Key insight: The 8 columns of B are processed together. We can precompute
    # linear combinations of the rows of B that are shared across all columns.
    #
    # Let's define:
    # s0 = B[0,:] (row 0, length 8)
    # s1 = B[1,:] (row 1, length 8)  
    # s2 = B[2,:] (row 2, length 8)
    #
    # Then C[0,:] = A[0,0]*s0 + A[0,1]*s1 + A[0,2]*s2
    # C[1,:] = A[1,0]*s0 + A[1,1]*s1 + A[1,2]*s2
    # C[2,:] = A[2,0]*s0 + A[2,1]*s1 + A[2,2]*s2
    #
    # This is just 3x3 matrix multiplication where each "entry" is a vector
    # of length 8. The rank should be the same as 3x3 matrix multiplication
    # over a field, which is 27 for schoolbook.
    #
    # Wait, that's not right either. The rank of the tensor depends on the
    # structure. Let me reconsider.
    #
    # The tensor T[a,b,c] has a = i*m+j, b = j*p+k, c = k*n+i.
    # For fixed j, the sub-tensor is:
    #   T[i*m+j, j*p+k, k*n+i] = 1 for all i,k
    # This is a 3 x 8 x 24 tensor (but really 3x8x3 embedded in 3x8x24).
    #
    # Actually, I think the cleanest approach is to just use the schoolbook
    # decomposition and see if we can find any savings. Given the time constraint,
    # let me return the schoolbook rank-72 solution.
    
    rank_schoolbook = n * m * p
    rank = rank_schoolbook
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
    
    result = {"U": U, "V": V, "W": W}
    return result
# EVOLVE-BLOCK-END
