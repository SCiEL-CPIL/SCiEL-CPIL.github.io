"""Matrix multiplication 2x4x5 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # The matrix multiplication tensor T_{i,j,k} = delta for C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]  (with W indexed as W[k*n+i, r])
    # Actually the standard form is: C[i,k] = sum_r U[i,j,r]... 
    # Let me be more careful.
    # 
    # Standard bilinear algorithm: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (coefficient)
    # Tensor decomposition: T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # But actually the multiplication tensor has entries T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # 
    # More precisely, for matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T with T[i,j,k] = 1 for all valid i,j,k (since each product contributes)
    # Wait, no. The tensor encodes: each term A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 means the (i,j,k) trilinear form is present.
    # 
    # A rank-r decomposition means: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # 
    # The constraint is: for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p)
    
    # Schoolbook rank is n*m*p = 2*4*5 = 40
    
    # Let's try to find a lower rank decomposition.
    # The lower bound is max(n*m, m*p, n*p) = max(8, 20, 10) = 20
    
    # Let's attempt to find a rank-20 decomposition or lower.
    # 
    # One approach: use the fact that we can group computations.
    # For n=2, m=4, p=5:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    # 
    # We need 20 outputs (C[0,0..4] and C[1,0..4]), each requiring 4 multiplications.
    # Total 80 multiplications in schoolbook, but tensor rank counts distinct products.
    
    # Let's try a systematic search for low-rank decompositions using linear algebra.
    # 
    # The tensor T can be viewed as a matrix of size (n*m) x (m*p) for each "slice" k... 
    # Actually, let's think of it differently.
    # 
    # Reshape: For fixed i, the map from (j,k) to C[i,k] is linear in B given A.
    # 
    # Let me try a direct approach: assume rank r, and see if we can find U, V, W.
    # 
    # The condition is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # This is a system of n*m*p = 40 equations in r * (n*m + m*p + p*n) = r * 38 unknowns.
    # For r = 20: 760 unknowns, 40 equations. Very underdetermined.
    # 
    # But the equations are nonlinear (products of U, V, W entries).
    # 
    # Let's try a different approach: fix U and V, then W is determined linearly.
    # Or use the fact that we want integer/half-integer solutions.
    
    # Let's try to construct a rank-20 decomposition.
    # 
    # Key insight: If we can make the tensor have a "separable" structure in some way.
    # 
    # Alternative: Use the fact that n=2 is small. We can potentially exploit the 2 rows.
    # 
    # Let's try: Can we do it with rank = m*p = 20?
    # If rank = m*p = 20, then V is a 20x20 matrix (m*p x rank).
    # 
    # Let's try: set V to be the identity (or a permutation), then:
    # sum_r U[i*m+j, r] * delta_{j*p+k, r} * W[k*n+i, r] = 1
    # => U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    # 
    # So for all i,j,k: U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    # 
    # This means: for each (j,k), we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1 for all i.
    # 
    # Let U[i*m+j, j*p+k] = a_{j,k,i} and W[k*n+i, j*p+k] = b_{j,k,i}
    # We need a_{j,k,i} * b_{j,k,i} = 1 for all i,j,k.
    # 
    # But U[i*m+j, j*p+k] is a single entry of U, shared across k for fixed i,j!
    # Wait, no: U[i*m+j, r] where r = j*p+k. So for fixed i,j, as k varies, r varies.
    # So U[i*m+j, j*p+k] for different k are different entries of U.
    # Similarly W[k*n+i, j*p+k] for different j are different entries of W.
    # 
    # So the constraint is: for all i in [0,2), j in [0,4), k in [0,5):
    #   U[i*4+j, j*5+k] * W[k*2+i, j*5+k] = 1
    # 
    # Let's denote u_{i,j,k} = U[i*4+j, j*5+k] and w_{i,j,k} = W[k*2+i, j*5+k]
    # We need u_{i,j,k} * w_{i,j,k} = 1 for all i,j,k.
    # 
    # But u_{i,j,k} is U[i*4+j, j*5+k]. For fixed i,j, this is a vector over k.
    # For fixed i,k, this is a vector over j (but with different column indices).
    # 
    # The issue: U[i*4+j, j*5+k] - the column index depends on j!
    # So U[i*4+j, :] has entries at columns j*5, j*5+1, ..., j*5+4 that are "active" for this j.
    # 
    # Actually, for a given row i*4+j of U, only the entries in columns j*5 through j*5+4 matter
    # (because V is identity, so only V[j*5+k, j*5+k] = 1 contributes).
    # 
    # So U[i*4+j, j*5+k] can be any value, and W[k*2+i, j*5+k] = 1/U[i*4+j, j*5+k].
    # 
    # But W[k*2+i, j*5+k] - for fixed k,i, as j varies, the column j*5+k varies.
    # So W[k*2+i, :] has entries at various columns.
    # 
    # The problem: W[k*2+i, j*5+k] must be consistent. For fixed k,i, different j give different columns.
    # So there's no conflict! Each (k,i,j) triple maps to a unique column j*5+k in W.
    # 
    # Similarly for U: for fixed i,j, different k give different columns j*5+k.
    # 
    # So we can set U[i*4+j, j*5+k] = 1 for all i,j,k, and W[k*2+i, j*5+k] = 1 for all i,j,k.
    # Then V is the identity.
    # 
    # But wait: is V the identity a valid 20x20 matrix? m*p = 20, rank = 20. Yes!
    # 
    # Let's verify: V[j*5+k, r] = delta_{j*5+k, r} (identity matrix).
    # U[i*4+j, r] = 1 if r is in [j*5, j*5+4], else 0.
    # W[k*2+i, r] = 1 if r is in [j*5, j*5+4] for some j... wait, no.
    # 
    # W[k*2+i, j*5+k] = 1 for all j. So W[k*2+i, r] = 1 if r = j*5+k for some j in [0,4).
    # That is, r mod 5 = k, and r < 20.
    # 
    # Let's check: sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*2+i, r]
    # = sum_r U[i*4+j, r] * delta_{j*5+k, r} * W[k*2+i, r]
    # = U[i*4+j, j*5+k] * W[k*2+i, j*5+k]
    # = 1 * 1 = 1. 
    # 
    # This works! Rank = 20.
    
    # But can we do better? The lower bound is max(n*m, m*p, n*p) = max(8, 20, 10) = 20.
    # So rank 20 is optimal!
    
    # Let's construct the rank-20 decomposition.
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # V is the identity matrix (m*p x rank = 20 x 20)
    for r in range(rank):
        V[r][r] = 1
    
    # U[i*m+j, r] = 1 if r is in [j*p, j*p+p-1] = [j*5, j*5+4]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = j * p + k
                U[i * m + j][r] = 1
    
    # W[k*n+i, r] = 1 if r = j*p+k for some j, i.e., r % p == k
    for i in range(n):
        for k in range(p):
            for j in range(m):
                r = j * p + k
                W[k * n + i][r] = 1
    
    # Verify the decomposition
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    # Fallback to schoolbook if verification fails
                    return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
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
