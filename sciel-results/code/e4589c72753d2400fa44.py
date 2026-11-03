"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # The matrix multiplication tensor T_{n,m,p}[i,j,k] = 1 if (i,j,k) is a valid index
    # We want to decompose this into rank r factors U, V, W such that:
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # T = sum_c U[:,c] \otimes V[:,c] \otimes W[:,c]
    # In index form: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For 3x4x8, the schoolbook rank is 3*4*8 = 96.
    # We can potentially do better with Strassen-like methods, but for simplicity
    # and correctness, let's first try to find a decomposition.
    
    # Actually, let's think about this differently.
    # The tensor is 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
    # The rank of this tensor (as a bilinear map) is the matrix multiplication rank.
    
    # For small cases, we can try to use known results or do a search.
    # But 3x4x8 is small enough that we might be able to find a good decomposition.
    
    # Let's start with schoolbook (rank 96) and see if we can improve.
    # Actually, the current seed is rank 96 with score 1.0.
    # We need to beat that.
    
    # Let's try a simple approach: use the fact that the tensor is a product of 3 vectors
    # in some sense? No, that's not right.
    
    # Let's think about the structure. The tensor T[i,j,k] = 1 for all i,j,k.
    # This is actually the "all-ones" tensor of size n x m x p.
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]B[j,k]
    # involves A[i,j] and B[j,k]. So T is 1 for all (i,j,k).
    
    # So we have the all-ones tensor of shape (n*m, m*p, p*n) with specific index mappings.
    # U has indices (i*m+j), V has indices (j*p+k), W has indices (k*n+i).
    
    # The rank of the all-ones tensor of shape (a,b,c) is 1 (it's rank 1).
    # But here the index mappings are different. Let's re-examine.
    
    # Actually, T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # This means the tensor is 1 everywhere in its domain.
    # So it's the all-ones tensor of shape (n*m, m*p, p*n).
    # The rank of an all-ones tensor is 1!
    
    # Wait, that can't be right. Let me double-check.
    # If T = U \otimes V \otimes W where U, V, W are vectors of all 1s, then
    # T[i*m+j, j*p+k, k*n+i] = U[i*m+j] * V[j*p+k] * W[k*n+i] = 1*1*1 = 1.
    # So yes, rank 1!
    
    # But that seems too good to be true. Let me re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 8 matrix."
    
    # Oh wait, I think I'm confusing the tensor definition. Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # So the tensor is defined such that:
    # C[k*n+i] = sum_{j} sum_{c} A[i*m+j] * B[j*p+k] * ... 
    # Actually, the standard definition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In the tensor formulation:
    # C[k*n+i] = sum_{j,c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # And we want this to equal sum_j A[i*m+j] * B[j*p+k]
    
    # So the tensor T[i*m+j, j*p+k, k*n+i] should be 1 if this term contributes to C[k*n+i],
    # which is when the indices are consistent: i is the row of A, j is the column of A and row of B,
    # k is the column of B. And C[k*n+i] is the element at row i, column k of C.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # This is indeed the all-ones tensor in the (i*m+j, j*p+k, k*n+i) indexing.
    
    # But wait, the all-ones tensor of shape (12, 32, 24) has rank 1.
    # So U, V, W would be all-ones vectors of length 12, 32, 24 respectively, and rank = 1.
    
    # Let me verify: if U is all 1s of length 12, V is all 1s of length 32, W is all 1s of length 24,
    # then sum_c U[a,c]*V[b,c]*W[d,c] = U[a,0]*V[b,0]*W[d,0] = 1*1*1 = 1.
    # And we need this to equal T[a,b,d] where a=i*m+j, b=j*p+k, d=k*n+i.
    # T[a,b,d] = 1 for all valid (i,j,k).
    
    # But is T really 1 for all (a,b,d)? No! T is only defined for (a,b,d) that correspond to valid (i,j,k).
    # For (a,b,d) that don't correspond to valid (i,j,k), T is 0 (or undefined).
    
    # So the tensor is 1 on the set of (a,b,d) = (i*m+j, j*p+k, k*n+i) for valid i,j,k,
    # and 0 elsewhere.
    
    # This is NOT the all-ones tensor. It's a sparse tensor with 1s only on a specific subset.
    
    # Let me reconsider. The rank of this tensor is the matrix multiplication rank, which is known to be
    # at most n*m*p = 96 for 3x4x8.
    
    # So my earlier analysis was wrong. The tensor is not all-ones; it's 1 only on the "consistent" indices.
    
    # Let's go back to schoolbook and see if we can do better.
    # For now, let's implement a correct schoolbook decomposition and see if the evaluator accepts it.
    # Then we can try to improve.
    
    deadline = time.time() + 100  # safety margin
    
    # Schoolbook decomposition
    rank = n * m * p  # 96
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
