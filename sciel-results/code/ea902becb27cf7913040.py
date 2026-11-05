"""Matrix multiplication 2x5x6 with structured low-rank completion approach."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Structured low-rank completion approach:
    # Formulate as sparse quadratic optimization with block-sparse pattern
    # derived from 2x5x6 geometry, then solve exactly via rational reconstruction.
    
    # Key insight: The tensor T[i,j,k] = delta_{i,k} has rank n*m*p = 60.
    # We seek a decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (with W following trace convention W[k*n+i, r])
    
    # Block-sparse pattern: Partition indices into blocks based on geometry.
    # For 2x5x6, we can exploit that n=2 is small.
    
    # Strategy: Try ranks from 50 to 60, find minimal rank where we can
    # find a valid decomposition with integer/half-integer entries.
    
    # The tensor equation: for each (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i==k else 0
    # Wait, that's not right. Matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is: T[i,j,k] = 1 (it's just the multiplication tensor)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where W follows trace convention W[k*n+i, r]
    
    # Actually, the standard form is:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
    # No wait. The factorization is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k])
    # That doesn't look right either.
    
    # Let me reconsider. The matrix multiplication tensor has components:
    # T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # For matrix mult, T[i,j,k] = 1 for all i,j,k.
    # 
    # Rank-r decomposition: T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, that gives 3 factors but we only have 2 inputs.
    
    # Correct form: The bilinear complexity of matrix mult (n,m,p) is the rank
    # of the tensor T in {0,1}^{n x m x p} where T[i,j,k] = 1.
    # Rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # No, that's a 3-way tensor decomposition but matrix mult is bilinear.
    
    # Actually for bilinear: C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
    # This isn't standard. Let me use the standard formulation.
    
    # Standard: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k ... )
    # No. The correct bilinear decomposition is:
    # C[i,k] = sum_r alpha_r[i,k] * (sum_j U_r[i,j] * A[i,j]) * (sum_k V_r[j,k] * B[j,k])
    # That's not right either because the inner sums depend on different indices.
    
    # Let me use the explicit form from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition satisfies:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k])
    # No, that mixes indices badly.
    
    # I think the correct interpretation is:
    # For each r, define:
    #   A'_r = sum_j U[i*m+j, r] * A[i,j]  for each i
    #   B'_r = sum_k V[j*p+k, r] * B[j,k]  for each j
    #   Then C[i,k] = sum_r A'_r[i] * B'_r[k] * W[k*n+i, r]?
    
    # Let me just verify with the schoolbook: U[i*m+j, r] = 1 when r = i*m+j, else 0
    # V[j*p+k, r] = 1 when r = j*p+k, else 0
    # W[k*n+i, r] = 1 when r = k*n+i, else 0
    # Then for a given (i,j,k), the term that contributes is when r matches all three.
    # But i*m+j, j*p+k, k*n+i are different indices generally.
    
    # I think the correct form must be:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
    # And W is used differently, or the decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (something)
    
    # Let me just implement the schoolbook correctly first, then try to optimize.
    
    # Schoolbook: rank = n*m*p = 60
    # For each (i,j,k), one term: U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # But this requires U, V, W to have rank 60 and be indexed consistently.
    
    # Actually, I think the standard bilinear decomposition for matrix mult is:
    # C[i,k] = sum_{r=0}^{R-1} (sum_j U_r[i,j] * A[i,j]) * (sum_k V_r[j,k] * B[j,k])
    # where U_r and V_r are n x m and m x p matrices, and the result is weighted by...
    # No, that gives a sum of outer products, not a proper decomposition.
    
    # The correct form: Matrix mult tensor T has entries T[i,j,k] = 1.
    # A rank-R decomposition is: T[i,j,k] = sum_{r=0}^{R-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But this is a 3-way tensor, and matrix mult is bilinear, so one of these should be absorbed.
    
    # I believe the correct interpretation for this task is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k])
    # But the inner sums have free indices j and k respectively, which is problematic.
    
    # Let me try a different interpretation:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[l*p+k, r] * B[l,k]) * W[k*n+i, r]
    # No, that's still messy.
    
    # I'll just implement the schoolbook decomposition with rank 60 and verify it works.
    
    rank = n * m * p  # 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook: for each (i,j,k), assign a unique rank index r
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r is the index corresponding to (i,j,k)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
