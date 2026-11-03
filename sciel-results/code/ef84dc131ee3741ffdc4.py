"""Program for rank-minimal tensor decomposition of 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = time.time() + 110  # safety margin
    
    # The tensor for matrix multiplication C = A * B where A is n x m and B is m x p
    # has entries T[i, j, k] such that C[i, k] = sum_j A[i, j] * B[j, k]
    # The rank-1 decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need to find the minimum rank R and matrices U (n*m x R), V (m*p x R), W (p*n x R)
    
    # For M_{2,3,6}: n=2, m=3, p=6
    # U shape: (6, R)
    # V shape: (18, R) 
    # W shape: (12, R)
    
    # Schoolbook rank = n*m*p = 36
    # We need to find a better rank.
    
    # Key insight: We can use the structure of the problem.
    # C is 2x6, A is 2x3, B is 3x6.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # Let's think about this as two independent 2x3 by 3x3 problems if we split B into two 3x3 blocks.
    # B = [B1, B2] where B1, B2 are 3x3
    # C = [C1, C2] where C1 = A*B1, C2 = A*B2
    # Each 2x3 by 3x3 multiplication has rank 15 (known result: rank(M_{2,3,3}) = 15)
    # So total rank would be 30. But we can do better.
    
    # Actually, let me reconsider. The known rank of M_{2,3,3} is 15.
    # But for M_{2,3,6}, we can potentially do better than 30.
    
    # Let me try a different approach: use the fact that we only have 2 rows in A.
    # A = [a00, a01, a02; a10, a11, a12]
    # B = 3x6 matrix
    # C = [c00, c01, c02, c03, c04, c05; c10, c11, c12, c13, c14, c15]
    
    # c0k = a00*B0k + a01*B1k + a02*B2k
    # c1k = a10*B0k + a11*B1k + a12*B2k
    
    # For each column k, we have 2 outputs that depend on 3 input products.
    # This is like a 2x3 matrix-vector multiplication, which has rank 2 (for each k independently).
    # But we have 6 columns, so naive rank = 12.
    # However, the same A entries are used across all columns, so we can share.
    
    # Actually, the bilinear complexity is the rank of the tensor.
    # Let me try to construct a rank-18 decomposition.
    
    # Strategy: For each pair of columns (k, k+3), use 3 rank-1 terms.
    # Total: 3 pairs * 3 terms = 9 terms? No, that's too few.
    
    # Let me think more carefully.
    # The tensor has 2*3*6 = 36 non-zero entries in the "full" representation.
    # Each rank-1 term can cover at most one (i,j,k) combination if we use the standard basis.
    # But with non-trivial U, V, W entries, one term can contribute to multiple C entries.
    
    # Let me try a systematic approach using known results.
    # The rank of M_{n,m,p} is at least max(n*m, m*p, p*n) = max(6, 18, 12) = 18.
    # So rank >= 18.
    # Can we achieve rank 18?
    
    # Construction for rank 18:
    # We need 18 rank-1 tensors that span the multiplication.
    
    # Let's use the following construction:
    # For r = 0, ..., 17:
    #   Let r = 3*q + s where q in {0,1,2} and s in {0,1,2}
    #   ... this doesn't seem right.
    
    # Alternative: use the fact that M_{2,3,6} can be decomposed as:
    # M_{2,3,6} = M_{2,3,3} + M_{2,3,3} (two independent 3-column blocks)
    # But this gives rank 30, not 18.
    
    # Let me try a different decomposition.
    # Consider that we can write:
    # C = A * B
    # Let's partition B into 3 pairs of columns: (0,3), (1,4), (2,5)
    # For each pair, we have 2 columns of C that depend on all 3 rows of B.
    
    # Actually, I think the minimum rank might be higher than 18.
    # Let me try rank 27 first, which is 3 * 9.
    
    # Wait, let me reconsider the lower bound.
    # rank(M_{n,m,p}) >= max(n*m, m*p, n*p)
    # For (2,3,6): max(6, 18, 12) = 18
    # So rank >= 18.
    
    # Let me try to construct a rank-18 decomposition using a different method.
    
    # Method: Use the "Kronecker product" structure.
    # M_{2,3,6} can be seen as M_{2,3,2} tensor M_{1,1,3} in some sense?
    # Not exactly.
    
    # Let me try a direct construction.
    # I'll use 18 terms, grouped as follows:
    # For each j in {0,1,2} and each k in {0,1,2}, define 3 terms.
    # Total: 3*3*3 = 27 terms. That's too many.
    
    # For each j in {0,1,2} and each pair of columns (k, k+3) for k in {0,1,2}:
    # Define 2 terms. Total: 3*3*2 = 18 terms.
    
    # Let me try this construction:
    # Term (j, k, t) for j in 0..2, k in 0..2, t in {0,1}:
    #   U[j + 3*t, ...] = 1  (selects A[0,j] for t=0, A[1,j] for t=1)
    #   V[j*p + k + 3*t, ...] = 1  (selects B[j,k] for t=0, B[j,k+3] for t=1)
    #   W[...] = ...
    
    # This is getting complex. Let me just implement the schoolbook rank 36 for now,
    # and note that the optimal rank is likely between 18 and 36.
    
    # Actually, let me try rank 18 with a specific construction.
    
    rank = 18
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Construction:
    # For each j in 0..2 (middle dimension), and for each k_pair in 0..2 (column pairs):
    #   We have 2 terms, one for row 0 of A, one for row 1 of A.
    #   Term index r = 6*j + 2*k_pair + t where t in {0,1}
    
    # Wait, that gives 3*3*2 = 18 terms.
    
    # Let me define:
    # For j in 0..2, k in 0..2, t in {0,1}:
    #   r = 6*j + 2*k + t
    #   U[0*m + j, r] = 1 if t == 0, else 0  (A[0,j] for t=0)
    #   U[1*m + j, r] = 0 if t == 0, else 1  (A[1,j] for t=1)
    #   V[j*p + k, r] = 1 if t == 0, else 0  (B[j,k] for t=0)
    #   V[j*p + (k+3), r] = 0 if t == 0, else 1  (B[j,k+3] for t=1)
    #   W[k*n + 0, r] = 1 if t == 0, else 0  (C[0,k] for t=0)
    #   W[(k+3)*n + 0, r] = 0 if t == 0, else 1  (C[0,k+3] for t=1)
    #   W[k*n + 1, r] = 0 if t == 0, else 1  (C[1,k] for t=0)
    #   W[(k+3)*n + 1, r] = 1 if t == 0, else 0  (C[1,k+3] for t=1)
    
    # Let me verify this construction:
    # For a given (i, j, k), the contribution to C[i,k] from term r is:
    #   U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    # For term r = 6*j' + 2*k' + t:
    #   U[i*m + j, r] is non-zero only if j == j' and i == t
    #   V[j*p + k, r] is non-zero only if j == j' and (k == k' if t==0, or k == k'+3 if t==1)
    #   W[k*n + i, r] is non-zero only if (k == k' and i == 0 if t==0, or k == k'+3 and i == 1 if t==1)
    
    # This doesn't seem to work correctly. Let me reconsider.
    
    # I think the construction is flawed. Let me just use the schoolbook decomposition
    # which is guaranteed to be correct, with rank 36.
    
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
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
