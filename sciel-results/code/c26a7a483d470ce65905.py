"""Program for finding a rank-minimal exact tensor decomposition for 2x4*4x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    We use a block-decomposition approach:
    - Split the middle dimension (m=4) into two blocks of size 2.
    - Each block corresponds to a 2x2 * 2x5 multiplication.
    - Find a low-rank decomposition for 2x2 * 2x5.
    - Combine the two blocks.
    
    The 2x2 * 2x5 tensor has 2*2*5 = 20 entries.
    Schoolbook rank is 20. We aim for rank 18 or lower per block.
    Total rank would be 2 * block_rank.
    
    If block_rank = 18, total rank = 36.
    If block_rank = 17, total rank = 34.
    If block_rank = 16, total rank = 32.
    
    We'll use numerical optimization to find a valid decomposition for the 2x2*2x5 block,
    then verify and snap to half-integers if possible.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # First, let's try to find a good decomposition for 2x2 * 2x5
    # The tensor T[i, j, k] for i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k])
    
    # Wait, the tensor for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor elements are T[i,j,k] = 1 for all valid i,j,k.
    # This is the "all-ones" tensor!
    
    # For the all-ones tensor of size n*m*p, the rank is 1!
    # U[i,j,c] = 1, V[j,k,c] = 1, W[k,i,c] = 1 for c=0 only.
    
    # But wait, the seed program returns rank 40, which suggests the tensor is NOT all-ones.
    # Let me re-read the task.
    
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This is just about the indexing of W.
    
    # The tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p) is:
    # T[i, j, k] = 1 if we consider the bilinear form.
    # Actually, the tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a 3-tensor with indices (i, j, k) where T[i,j,k] = 1.
    
    # So the tensor IS all-ones, and rank 1 should work!
    
    # Let me verify: if rank = 1, U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1*1*1 = 1 = T[i,j,k].
    
    # This works! So rank 1 is optimal.
    
    # But the seed has rank 40, which is the schoolbook rank. This suggests that
    # the evaluator might be checking something else, or the tensor is not all-ones.
    
    # Let me re-read: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix."
    
    # I think the issue is that the "tensor decomposition" here refers to the bilinear complexity,
    # which is the rank of the matrix multiplication tensor. For the all-ones tensor, the rank is 1.
    
    # However, in the context of matrix multiplication algorithms, the "rank" usually refers to
    # the number of scalar multiplications, which for the all-ones tensor would be 1.
    
    # But that can't be right, because then every matrix multiplication would have rank 1.
    
    # I think I'm confusing the tensor with the bilinear map. Let me reconsider.
    
    # The matrix multiplication tensor T is an element of (R^n*)^⊗(R^m*)^⊗(R^p*)^⊗...
    # No, that's not right either.
    
    # The standard way: the multiplication tensor for (n,m,p) is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    # The element T[i,j,k] is 1 if the indices correspond to a valid multiplication term.
    
    # For C[i,k] = sum_j A[i,j] B[j,k], the tensor element T[(i,j), (j,k), (k,i)] = 1.
    # In flattened indices: U index = i*m + j, V index = j*p + k, W index = k*n + i.
    # T[U_idx, V_idx, W_idx] = 1 if U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i for some i,j,k.
    
    # So T is NOT all-ones. T is a sparse tensor with 1s at specific positions.
    # Specifically, T has 1s at positions where the three indices are "compatible".
    
    # For n=2, m=4, p=5:
    # T[i*4+j, j*5+k, k*2+i] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}.
    # All other entries are 0.
    
    # This is the correct tensor. The schoolbook decomposition uses 40 terms, one for each (i,j,k).
    
    # Now, can we do better? The rank of this tensor is what we're looking for.
    
    # Let's use the block decomposition: split j into two groups of 2.
    # Block 1: j in {0,1}, Block 2: j in {2,3}.
    # Each block is a 2x2 * 2x5 tensor.
    
    # For the 2x2 * 2x5 tensor, the schoolbook rank is 2*2*5 = 20.
    # Can we do better?
    
    # Let's try to find a rank-18 or rank-17 decomposition for 2x2 * 2x5.
    
    def find_block_decomposition(block_rank):
        """Find a rank-block_rank decomposition for 2x2 * 2x5."""
        bn, bm, bp = 2, 2, 5
        U = np.zeros((bn * bm, block_rank))
        V = np.zeros((bm * bp, block_rank))
        W = np.zeros((bp * bn, block_rank))
        
        # Initialize with random values
        np.random.seed(42)
        for attempt in range(100):
            U[:] = np.random.randn(bn * bm, block_rank) * 0.1
            V[:] = np.random.randn(bm * bp, block_rank) * 0.1
            W[:] = np.random.randn(bp * bn, block_rank) * 0.1
            
            # Compute the tensor from the factors
            T_computed = np.zeros((bn, bm, bp))
            for i in range(bn):
                for j in range(bm):
                    for k in range(bp):
                        T_computed[i, j, k] = np.sum(
                            U[i*bm + j, :] * V[j*bp + k, :] * W[k*bn + i, :]
                        )
            
            # Target tensor: T[i,j,k] = 1 for all i,j,k
            T_target = np.ones((bn, bm, bp))
            
            # Check if the decomposition is valid
            if np.allclose(T_computed, T_target, atol=1e-6):
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                T_snapped = np.zeros((bn, bm, bp))
                for i in range(bn):
                    for j in range(bm):
                        for k in range(bp):
                            T_snapped[i, j, k] = np.sum(
                                U_snap[i*bm + j, :] * V_snap[j*bp + k, :] * W_snap[k*bn + i, :]
                            )
                
                if np.allclose(T_snapped, T_target, atol=1e-6):
                    return U_snap, V_snap, W_snap
                
                # If snapping doesn't work, use the original
                return U, V, W
        
        return None
    
    # Try different block ranks
    for block_rank in [18, 19, 20]:
        U1, V1, W1 = find_block_decomposition(block_rank)
        if U1 is not None:
            total_rank = 2 * block_rank
            U_total = np.zeros((n * m, total_rank))
            V_total = np.zeros((m * p, total_rank))
            W_total = np.zeros((p * n, total_rank))
            
            # Block 1 (j=0,1) uses first half of columns
            U_total[0:4, 0:block_rank] = U1
            V_total[0:10, 0:block_rank] = V1
            W_total[0:10, 0:block_rank] = W1
            
            # Block 2 (j=2,3) uses second half of columns
            U_total[4:8, block_rank:total_rank] = U1
            V_total[10:20, block_rank:total_rank] = V1
            W_total[0:10, block_rank:total_rank] = W1
            
            return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    
    # Fallback: Rank 40 schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
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
