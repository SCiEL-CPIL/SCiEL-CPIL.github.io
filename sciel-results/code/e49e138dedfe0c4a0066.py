"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Formulate as a linear code problem:
    # Output C is in F^{p*n} = F^{18}
    # Input A is in F^{n*m} = F^{6}, Input B is in F^{m*p} = F^{27}
    # C = A * B means C[k*n + i] = sum_j A[i*m + j] * B[j*p + k]
    # 
    # The multiplication tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We need T = sum_r U_r outer V_r outer W_r
    # 
    # Key insight: For fixed i (output row), C[i,:] = A[i,:] * B (2x3 times 3x9)
    # This is a linear map from B to C[i,:] parameterized by A[i,:]
    # 
    # View each output element C[k*n+i] as a bilinear form in (A, B).
    # The space of bilinear forms has dimension (n*m)*(m*p) = 6*27 = 162
    # But our specific bilinear forms are structured.
    #
    # Approach: Build the 3-way tensor and find its minimal rank decomposition
    # using the structure that for each (i,k) pair, the bilinear form is
    # sum_j A[i*m+j] * B[j*p+k]
    #
    # For 2x3x9: We have 18 output elements, each is a sum of 3 products.
    # Total "schoolbook" terms = 2*3*9 = 54
    #
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    #
    # Let's try to achieve rank 27 by finding 27 bilinear forms that span the output.
    #
    # Strategy: Use the fact that for each k (0..8), we need to compute
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # This is a 2x3 matrix times a 3x1 vector, which can be done with rank 3
    # (one per j). So naively 9*3 = 27 terms.
    #
    # But can we share across k? The key is that B[j,k] appears in both
    # C[0,k] and C[1,k] for the same j,k.
    #
    # Let's try rank = 27 with the following construction:
    # For each j (0,1,2) and each k (0..8), create a term that uses A[i,j] and B[j,k]
    # to contribute to C[i,k] for i=0,1.
    #
    # Term r = j*9 + k:
    #   U[i*m + j, r] = 1 for i=0,1  (i.e., U[j, r]=1, U[3+j, r]=1)
    #   V[j*p + k, r] = 1
    #   W[k*n + i, r] = 1 for i=0,1  (i.e., W[k*2, r]=1, W[k*2+1, r]=1)
    #
    # Check: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For a=i*m+j, b=j'*p+k', c=k''*n+i'':
    #   Term r=j*9+k contributes if j'=j, k'=k, and i''=i (from W[k*2+i'', r]=1)
    #   So U[i*m+j, r]=1 requires the i in a to match... 
    #   Wait, U[a,r] where a=i*m+j. For term r=j*9+k, U[a,r]=1 when a=j (i=0) or a=3+j (i=1)
    #   V[b,r]=1 when b=j*p+k
    #   W[c,r]=1 when c=2k (i''=0) or c=2k+1 (i''=1)
    #   
    #   So T[i*m+j, j*p+k, 2k+i''] = 1 if i'' matches the i from U... 
    #   Actually U[i*m+j, r]=1 for BOTH i=0 and i=1 when the j matches.
    #   So T[0*m+j, j*p+k, 2k+0] = 1 (i=0 in U, i''=0 in W) ✓
    #   T[0*m+j, j*p+k, 2k+1] = 1 (i=0 in U, i''=1 in W) -- but this should be 0!
    #   T[1*m+j, j*p+k, 2k+0] = 1 (i=1 in U, i''=0 in W) -- but this should be 0!
    #   T[1*m+j, j*p+k, 2k+1] = 1 (i=1 in U, i''=1 in W) ✓
    #
    # So this doesn't work directly because U and W are "entangled" across i.
    #
    # We need a different approach. Let's use the linear algebra method:
    # Fix the structure and solve for the factors.
    #
    # Alternative: Use the standard result that the rank of matrix multiplication
    # 2x3x9 is at most 27 (trivial) and try to verify if 27 is achievable.
    #
    # Let me try a different construction with rank 27:
    # For each (i,j) pair (6 pairs), we have A[i,j]. For each such pair,
    # we need to multiply by all of B[j,:] (9 elements) to get contributions to C[i,:].
    # 
    # Term for (i,j,k): standard schoolbook, rank 54.
    #
    # To reduce: Note that for fixed (i,j), the contribution to C[i,k] for all k
    # is A[i,j] * B[j,k]. This is a "row" of products.
    #
    # Let's try: for each j (3 values) and each linear combination of i (2 values),
    # we can potentially combine. But the structure is rigid.
    #
    # Let me just verify the schoolbook works and return it. Rank 54.
    # Then try to find if we can do better with a systematic search.
    
    # Actually, let's try a smarter approach using the tensor structure.
    # The tensor T has shape (6, 27, 18).
    # T[i*3+j, j*9+k, k*2+i] = 1
    
    # Let's try to find a rank-27 decomposition by solving a system.
    # Assume W is the identity-like structure and solve for U, V.
    
    # Better approach: use the fact that the problem is equivalent to finding
    # the tensor rank of the multiplication tensor. For 2x3x9, known results
    # suggest rank 27 might be achievable.
    
    # Let me try constructing rank 27 explicitly:
    # Use 27 terms, one for each (j, k) pair where j in 0..2, k in 0..8.
    # For term r = j*9 + k:
    #   We want it to contribute to C[0,k] via A[0,j]*B[j,k] and C[1,k] via A[1,j]*B[j,k]
    #   
    # Set: U[0*3+j, r] = 1, U[1*3+j, r] = 1  (both rows of A for this j)
    #      V[j*9+k, r] = 1
    #      W[k*2+0, r] = 1, W[k*2+1, r] = 1  (both outputs for this k)
    #
    # Check T[0*3+j, j*9+k, k*2+0]: U[0*3+j,r]=1, V[j*9+k,r]=1, W[k*2+0,r]=1 => 1 ✓
    # Check T[0*3+j, j*9+k, k*2+1]: U[0*3+j,r]=1, V[j*9+k,r]=1, W[k*2+1,r]=1 => 1 ✗ (should be 0)
    #
    # So this overcounts. The issue is that U and W both have "both i values" active.
    #
    # We need to separate the i dependence. Let's use a different basis.
    # For each (i, j, k), we need a separate term in the schoolbook.
    # 
    # To reduce rank, we can use linear combinations. Consider for fixed j,k:
    # C[0,k] += A[0,j]*B[j,k]
    # C[1,k] += A[1,j]*B[j,k]
    #
    # We can write this as:
    # C[0,k] += (A[0,j]+A[1,j])/2 * B[j,k] + (A[0,j]-A[1,j])/2 * B[j,k]
    # C[1,k] += (A[0,j]+A[1,j])/2 * B[j,k] - (A[0,j]-A[1,j])/2 * B[j,k]
    #
    # This still uses 2 terms per (j,k), so 2*3*9=54 total. No improvement.
    #
    # The rank 27 lower bound comes from the fact that we need at least 27 terms
    # because the B input has 27 dimensions and each term can only "use" one B element
    # effectively. But this bound may not be tight.
    #
    # Let me just return the schoolbook decomposition with rank 54.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
