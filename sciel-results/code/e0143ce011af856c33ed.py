DIMENSIONS = [2, 4, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition patterns
    # Indices for 2x2: i,j,k in {0,1}
    # U, V, W are 4x7 (flattened)
    
    # r=0
    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    
    # r=1
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    
    # r=2
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    
    # r=3
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    
    # r=4
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    
    # r=5
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    
    # r=6
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]
    
    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val
    
    # Sub-problem 1: A_L (i:0-1, j:0-1) * B_T (j:0-1, k:0-7)
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    
    # Sub-problem 2: A_R (i:0-1, j:2-3) * B_B (j:2-3, k:0-7)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    # Sparsification: find and eliminate linearly dependent rank terms
    # Build the combined tensor constraint matrix
    # The decomposition requires: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 if (i,j,k) is a valid multiplication index, else 0
    # 
    # We have 8 independent Strassen blocks. Check if any rank direction is linearly
    # dependent across blocks by examining the combined U, V, W matrices.
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to reduce rank by finding null space directions
    # The rank-r decomposition means we have r vectors in U, V, W spaces.
    # If some combination of rank directions gives zero contribution, we can eliminate it.
    
    # Check for linear dependencies by looking at the "tensor" formed by outer products
    # For each rank direction r, we have a rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # If sum_r c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0 for some nonzero c, we can reduce rank.
    
    # More practically: check if the column space of U has dimension < rank
    # Similarly for V and W. But the constraint is on the combined tensor.
    
    # Let's try a different approach: verify the decomposition is valid, then try
    # to find a lower-rank equivalent by solving a system.
    
    # First, verify current decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for valid (i,j,k)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(len(U_mat[0])):
                        total += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    # For matrix multiplication, the tensor T[i,j,k] = 1 always
                    # (since every (i,j,k) triple corresponds to a valid product term)
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None
    
    valid, err = verify(U, V, W, n, m, p)
    if not valid:
        # Decomposition is wrong, return as-is (shouldn't happen with correct Strassen)
        return {"U": U, "V": V, "W": W}
    
    # Now try to find linear dependencies among the rank directions
    # Build the "constraint matrix" C where C[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need C * 1 = 1 (all-ones vector) and want to find the minimum number of columns
    # that can still represent the target.
    
    # Actually, the target is the matrix multiplication tensor, which has a specific structure.
    # Let's build C and see its rank.
    
    num_ijk = n * m * p
    C = np.zeros((num_ijk, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k
                for r in range(rank):
                    C[idx, r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # The target is all ones (since T[i,j,k] = 1 for all valid triples in matmul)
    target = np.ones(num_ijk)
    
    # Check if C @ ones(rank) = target
    # This should be true by construction
    
    # Now, find the minimum subset of columns that still satisfies C_sub @ ones = target
    # This is a hard problem, but we can try:
    # 1. Find the rank of C
    # 2. If rank < number of columns, there's linear dependence
    
    # Compute SVD of C
    try:
        U_sv, s_vals, Vt = np.linalg.svd(C, full_matrices=False)
        # Find effective rank
        tol = 1e-10
        effective_rank = np.sum(s_vals > tol)
        
        if effective_rank < rank:
            # There are linear dependencies!
            # The null space of C has dimension rank - effective_rank
            # We can eliminate null space directions
            
            # Get null space basis
            null_space_dim = rank - effective_rank
            if null_space_dim > 0:
                # Null space vectors are the last null_space_dim rows of Vt
                null_basis = Vt[effective_rank:]
                
                # For each null space vector, we can add a multiple of it to the rank directions
                # without changing the decomposition. Use this to zero out entries.
                
                # Strategy: for each null space direction, find the rank index where it has
                # the largest component, and use it to eliminate that component from other ranks.
                
                # Actually, let's just check if we can reduce by 1
                # Try removing each rank direction and see if the remaining can still work
                
                # Simpler: just report the effective rank and try to construct a lower-rank solution
                # by keeping only the top effective_rank singular vectors
                # But this changes the actual U, V, W matrices...
                
                # The correct approach: if v is in null space of C, then for any scalar t,
                # adding t*v to the "rank direction" doesn't change C @ ones.
                # But we need to update U, V, W consistently.
                
                # This is complex. Let's just verify and return.
                pass
    except:
        pass
    
    # For now, return the verified decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
