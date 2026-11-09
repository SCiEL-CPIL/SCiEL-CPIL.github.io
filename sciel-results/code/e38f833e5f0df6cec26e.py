"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    
    The strategy is to use a Winograd-like frequency domain approach on the p=8 dimension,
    combined with a small amount of search to find a low-rank decomposition.
    
    The matrix multiplication tensor T[i,j,k] = delta (for A[i,j] * B[j,k] -> C[i,k])
    can be viewed as a set of 2*3 = 6 independent convolutions along the k-dimension
    (one for each (i,j) pair), but they share the B[j,k] factors.
    
    Actually, let's think more carefully. The tensor is T[i, j, k] where:
    - i in range(2)  (rows of A, rows of C)
    - j in range(3)  (cols of A, rows of B)
    - k in range(8)  (cols of B, cols of C)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a 3-mode tensor of size 2x3x8. The rank of this tensor is what we want to minimize.
    
    For a fixed j, we have 2*8 = 16 entries that all involve B[j,k]. 
    
    Let's try a different approach: we can view this as 3 independent matrix multiplications
    of size 2x1 times 1x8, but they share structure.
    
    Actually, the simplest non-trivial decomposition: for each j, the contribution is
    A[:,j] * B[j,:], which is an outer product of a 2-vector and an 8-vector.
    So we need at least 3 * (2*8) = 48 scalar multiplications in the schoolbook.
    
    But we can do better by combining. Let's try to find a decomposition of rank < 48.
    
    One approach: use the fact that we can combine the 3 'j' dimensions. 
    For each (i,k) pair, C[i,k] = sum_j A[i,j] B[j,k].
    
    Let's try a random search for a low-rank decomposition with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # The tensor T has entries T[i, j, k] = 1 if we're computing the product,
    # but actually the tensor for matrix multiplication is:
    # T[i, j, k] contributes to C[i, k] via A[i, j] * B[j, k]
    # So the tensor is just the "identity" in the sense that each (i,j,k) triple
    # corresponds to one multiplication.
    
    # The tensor we need to decompose: for each term in the sum
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the full tensor of the multiplication)
    # Wait, no. The tensor is defined such that:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # A rank-r decomposition means:
    # T[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right either. Let me re-read the problem.
    
    # The interface says:
    # U: (n*m, rank)
    # V: (m*p, rank)  
    # W: (p*n, rank)
    # And the decomposition is:
    # For the matrix multiplication tensor, we need:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # But T[i,j,k] = 1 for all i,j,k (since every product A[i,j]*B[j,k] contributes to C[i,k]).
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a 3-way tensor decomposition problem.
    
    # Let's verify with the schoolbook: rank = n*m*p = 48
    # For each (i,j,k), we have one term with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    
    # Can we do better? Let's try to find a decomposition of rank < 48.
    
    # One idea: group by j. For fixed j, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,k
    # This is like saying the "slice" for each j should give the all-ones 2x8 matrix
    # as a sum of rank-1 matrices.
    
    # The all-ones 2x8 matrix has rank 1, so for each j, we could potentially use just 1 term
    # if we could share across j's. But the U and V factors depend on j differently.
    
    # Let's try a structured approach:
    # Use the fact that 8 = 4*2, and try a 2-level decomposition.
    
    # Actually, let's just do a random search for a low-rank decomposition.
    # We'll try ranks from 24 upward (half the schoolbook) and see if we can find one.
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds margin
    
    # First, let's try a structured decomposition.
    # Idea: for each j, use a 2x8 all-ones matrix decomposition.
    # The 2x8 all-ones matrix can be written as u * v^T where u = [1,1]^T and v = [1,1,...,1]^T (8 ones).
    # So for each j, we need 1 "effective" multiplication, but the factors must be consistent.
    
    # Let's try: for each j, define:
    # U[i*m+j, c_j] = 1 for i=0,1
    # V[j*p+k, c_j] = 1 for k=0,...,7  
    # W[k*n+i, c_j] = 1 for all k,i
    # But this would require W to be all ones for each j, which means W[k*n+i, c_j] = 1 for all k,i
    # And we'd have 3 terms (one per j).
    
    # Check: sum_c U[i*m+j', c] * V[j'*p+k, c] * W[k*n+i, c]
    # For c = c_j: U[i*m+j', c_j] is 1 only if j'=j, V[j'*p+k, c_j] is 1 only if j'=j
    # So the product is 1 only when j'=j, and then it's 1*1*1 = 1
    # Sum over j' gives 1. 
    
    # Wait, this works! Let me verify more carefully.
    
    # We have 3 terms (one for each j).
    # For term c_j (j = 0, 1, 2):
    # U[i*m+j, c_j] = 1 for all i in {0,1}
    # V[j*p+k, c_j] = 1 for all k in {0,...,7}
    # W[k*n+i, c_j] = 1 for all k in {0,...,7}, i in {0,1}
    
    # Now check the sum: for given (i, j', k):
    # sum_{j=0}^{2} U[i*m+j', c_j] * V[j'*p+k, c_j] * W[k*n+i, c_j]
    # = sum_{j=0}^{2} [1 if j'=j else 0] * [1 if j'=j else 0] * 1
    # = 1 (when j'=j, only one term is nonzero)
    
    # This gives rank 3! But wait, that seems too good. Let me double-check.
    
    # Actually, I think I'm confusing myself. The tensor T[i,j,k] = 1 means that
    # every (i,j,k) triple contributes. The decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # With my proposed decomposition:
    # T[i,j,k] = sum_{c=0}^{2} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For c = j: U[i*m+j, j] = 1, V[j*p+k, j] = 1, W[k*n+i, j] = 1
    # For c != j: U[i*m+j, c] = 0 (since c != j, and U[i*m+j, c] is only nonzero when c=j)
    
    # So T[i,j,k] = 1 * 1 * 1 = 1. Correct!
    
    # So rank 3 works! But wait, is this valid? Let me re-examine.
    
    # The issue is that U has shape (n*m, rank) = (6, 3)
    # U[i*m+j, c] where i in {0,1}, j in {0,1,2}, c in {0,1,2}
    # U[0*3+0, 0] = U[0,0] = 1
    # U[1*3+0, 0] = U[3,0] = 1
    # U[0*3+1, 1] = U[1,1] = 1
    # U[1*3+1, 1] = U[4,1] = 1
    # U[0*3+2, 2] = U[2,2] = 1
    # U[1*3+2, 2] = U[5,2] = 1
    # All other entries are 0.
    
    # V has shape (m*p, rank) = (24, 3)
    # V[j*p+k, c] where j in {0,1,2}, k in {0,...,7}, c in {0,1,2}
    # V[0*8+k, 0] = 1 for all k
    # V[1*8+k, 1] = 1 for all k
    # V[2*8+k, 2] = 1 for all k
    # All other entries are 0.
    
    # W has shape (p*n, rank) = (16, 3)
    # W[k*n+i, c] where k in {0,...,7}, i in {0,1}, c in {0,1,2}
    # W[k*2+i, c] = 1 for all k, i, c
    # So W is all ones!
    
    # Let's verify: T[i,j,k] = sum_c U[i*3+j, c] * V[j*8+k, c] * W[k*2+i, c]
    # For (i=0, j=1, k=3):
    # c=0: U[0*3+1, 0] = U[1,0] = 0
    # c=1: U[0*3+1, 1] = U[1,1] = 1, V[1*8+3, 1] = V[11,1] = 1, W[3*2+0, 1] = W[6,1] = 1
    # c=2: U[0*3+1, 2] = U[1,2] = 0
    # Sum = 0 + 1*1*1 + 0 = 1. Correct!
    
    # So rank 3 is achievable! This is much better than 48.
    
    # But wait, can we do even better? Rank 1 or 2?
    # For rank 1: T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    # This would require U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k
    # This means U[i*m+j, 0] = a(i,j), V[j*p+k, 0] = b(j,k), W[k*n+i, 0] = c(k,i)
    # And a(i,j) * b(j,k) * c(k,i) = 1 for all i,j,k
    # Taking logs: log a(i,j) + log b(j,k) + log c(k,i) = 0
    # This is a system that may or may not have a solution.
    # For i=0,j=0,k=0: log a(0,0) + log b(0,0) + log c(0,0) = 0
    # For i=0,j=0,k=1: log a(0,0) + log b(0,1) + log c(1,0) = 0
    # Subtracting: log b(0,0) - log b(0,1) + log c(0,0) - log c(1,0) = 0
    # This imposes constraints. It's possible but not obvious.
    
    # Let's try rank 2 first, since it's more likely to work.
    
    # Actually, let's just return the rank-3 solution since it's clean and exact.
    # We can always try to improve later.
    
    rank = 3
    U = [[0] * rank for _ in range(n * m)]  # (6, 3)
    V = [[0] * rank for _ in range(m * p)]  # (24, 3)
    W = [[0] * rank for _ in range(p * n)]  # (16, 3)
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
