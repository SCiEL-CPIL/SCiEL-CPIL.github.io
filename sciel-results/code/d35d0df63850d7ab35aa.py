"""Matrix multiplication 2x3 x 3x9 tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Rank-18 decomposition: 3 independent 2x3x3 blocks, each rank 6
    # Using the known rank-6 decomposition for 2x3x3 matrix multiplication
    
    # For 2x3x3, we use a decomposition based on the structure:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a 2x3x3 tensor with rank at most 6 (schoolbook)
    # But we can do better: the rank of 2x3x3 is known to be 6
    
    # Let's construct a rank-6 decomposition for each 2x3x3 block
    # using a structured approach
    
    rank_per_block = 6
    num_blocks = p // m  # 3 blocks
    rank = num_blocks * rank_per_block  # 18
    
    U = [[0] * rank for _ in range(n * m)]   # shape (6, 18)
    V = [[0] * rank for _ in range(m * p)]   # shape (27, 18)
    W = [[0] * rank for _ in range(p * n)]   # shape (18, 18)
    
    # For each 2x3x3 block (output columns t*m to t*m+m-1),
    # we need a rank-6 decomposition of the 2x3x3 tensor.
    # 
    # The 2x3x3 tensor T[i,j,k] = delta[i,j,k] (the multiplication tensor)
    # has the property that for fixed i, the 3x3 matrix T[i,:,:] has rank 1.
    # 
    # A rank-6 decomposition: we can use the fact that
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # One approach: use 6 rank-1 tensors.
    # Let's use the following construction:
    # For each pair (i, j) where i in {0,1}, j in {0,1,2},
    # we get a rank-1 term: A[i,j] * B[j,k] for all k.
    # But that gives 6 terms, each contributing to all 3 output columns k.
    # 
    # Actually, the standard schoolbook for 2x3x3 uses 2*3*3 = 18 terms.
    # To get rank 6, we need a different approach.
    #
    # The key insight: for 2x3x3, we can use the decomposition:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum over 6 terms if we can factor the 3x3 part.
    #
    # Actually, let me think again. The tensor is T[i,j,k] = 1 if the 
    # multiplication A[i,j]*B[j,k] contributes to C[i,k].
    # 
    # For 2x3x3, the tensor has dimensions (2,3,3).
    # The rank of this tensor: since for each i, the slice T[i,:,:] is a 
    # 3x3 matrix of all 1s (rank 1), and there are 2 such slices,
    # the tensor can be written as sum of 2 rank-3 tensors? No.
    #
    # Let me use a concrete rank-6 decomposition:
    # Term r for r in 0..5:
    #   i = r // 3, j = r % 3
    #   U[i*m+j][r] = 1
    #   V[j*p + k][r] = 1 for all k in 0..2 (this doesn't work for rank-1)
    #
    # A rank-1 term must have U, V, W all with a single nonzero entry pattern
    # that factors. Let me use the following:
    #
    # For the 2x3x3 tensor, a rank-1 term (u, v, w) satisfies:
    #   T[i,j,k] = sum_r u[i,j,r] * v[j,k,r] * w[i,k,r]  (approximately)
    #
    # Actually, the tensor decomposition is:
    #   T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For a rank-1 term r: U[i*m+j,r] = a_{i,j}, V[j*p+k,r] = b_{j,k}, W[k*n+i,r] = c_{k,i}
    # We need: sum_r a_{i,j} b_{j,k} c_{k,i} = delta_{i,i} * 1 (i.e., 1 for all i,j,k)
    # Wait, T[i,j,k] = 1 for all i,j,k in the multiplication tensor? No.
    # 
    # The multiplication tensor for C = A*B where A is 2x3, B is 3x3:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
    # T is a 2x3x3 tensor where T[i,j,k] = 1 if j is the contracted index.
    # Actually T[i,j,k] = 1 for all valid (i,j,k) - it's the all-ones tensor!
    #
    # Wait no. The tensor is T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # This is just the all-ones 2x3x3 tensor.
    #
    # The rank of the all-ones 2x3x3 tensor: it can be written as 
    # sum_{j=0}^{2} (e_i * 1) * (1 * e_j) * (1 * e_k)? No.
    #
    # A rank-1 tensor is u_i * v_j * w_k.
    # The all-ones tensor = sum_{j=0}^{2} (1 for all i) * (delta_{j',j} for j') * (1 for all k)
    # = sum_{j=0}^{2} u^{(j)} v^{(j)} w^{(j)}
    # where u^{(j)}_i = 1, v^{(j)}_{j'} = delta_{j',j}, w^{(j)}_k = 1
    #
    # So rank = 3! Not 6.
    #
    # Let me verify: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j (sum_{r} U[i,j,r] V[j,k,r] W[k,i,r])
    #
    # If we use rank 3: for r = 0,1,2 corresponding to j = 0,1,2:
    #   U[i,j,r] = 1 if j == r, else 0  -> U[i*m+j, r] = delta_{j,r}
    #   V[j,k,r] = 1 if j == r, else 0  -> V[j*p+k, r] = delta_{j,r}
    #   W[k,i,r] = 1 for all k,i         -> W[k*n+i, r] = 1
    #
    # Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = sum_r delta_{j,r} delta_{j,r} * 1 = delta_{j,j} = 1
    # 
    # But wait: U[i*m+j, r] = delta_{j,r} means U depends on j, not i.
    # V[j*p+k, r] = delta_{j,r} means V depends on j, not k.
    # W[k*n+i, r] = 1 for all k, i.
    #
    # Product: delta_{j,r} * delta_{j,r} * 1 = delta_{j,r}
    # Sum over r: sum_r delta_{j,r} = 1 (since j is in {0,1,2})
    #
    # So T[i,j,k] = 1 for all i,j,k. Correct!
    #
    # So each 2x3x3 block has rank 3, not 6!
    # Total rank = 3 * 3 = 9.
    
    # Let me reconstruct with rank 9.
    rank_per_block = 3
    rank = num_blocks * rank_per_block  # 9
    
    U = [[0] * rank for _ in range(n * m)]   # shape (6, 9)
    V = [[0] * rank for _ in range(m * p)]   # shape (27, 9)
    W = [[0] * rank for _ in range(p * n)]   # shape (18, 9)
    
    col = 0
    for t in range(num_blocks):
        for j in range(m):  # j = 0, 1, 2
            r = col
            # U[i*m + j, r] = 1 for all i
            for i in range(n):
                U[i * m + j][r] = 1
            # V[j*p + k, r] = 1 for all k in the block
            for k in range(m):
                V[j * p + (t * m + k)][r] = 1
            # W[k*n + i, r] = 1 for all k in the block, all i
            for k in range(m):
                for i in range(n):
                    W[(t * m + k) * n + i][r] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
