"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    current_rank = 0
    
    # Group 1: A1 * Btop
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    U[i * m + j_loc][r] = u_loc[i * 2 + j_loc]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[j_loc * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    # Group 2: A2 * Bbottom
    for q in range(4):
        for s in range(7):
            u_loc, v_loc, w_loc = strassen_terms[s]
            r = current_rank
            for i in range(2):
                for j_loc in range(2):
                    U[i * m + (j_loc + 2)][r] = u_loc[i * 2 + j_loc]
            for j_loc in range(2):
                for k_loc in range(2):
                    V[(j_loc + 2) * p + (2 * q + k_loc)][r] = v_loc[j_loc * 2 + k_loc]
            for i in range(2):
                for k_loc in range(2):
                    W[(2 * q + k_loc) * n + i][r] = w_loc[k_loc * 2 + i]
            current_rank += 1
            
    # Now try to find linear dependencies among the 56 rank-1 tensors
    # Each rank-1 tensor is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Flatten each into a vector of size n*m*p = 64
    
    U_np = np.array(U, dtype=np.float64).T  # (56, 8)
    V_np = np.array(V, dtype=np.float64).T  # (56, 32)
    W_np = np.array(W, dtype=np.float64).T  # (56, 16)
    
    # Stack the 56 rank-1 tensors as vectors
    # Each tensor T_r has entries T_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
    # Flatten to vector of length n*m*p = 64
    # Index in flattened tensor: (i * m + j) * p + k = i*m*p + j*p + k
    
    tensor_stack = np.zeros((rank, n * m * p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_stack[r, i * m * p + j * p + k] = U_np[r, i * m + j] * V_np[r, j * p + k] * W_np[r, k * n + i]
    
    # Compute the rank of this 56 x 64 matrix
    # Use SVD to find the numerical rank
    try:
        U_svd, s, Vt = np.linalg.svd(tensor_stack, full_matrices=False)
        # Count singular values above tolerance
        tol = 1e-10
        effective_rank = np.sum(s > tol)
        
        if effective_rank < rank:
            # We found dependencies! Extract a basis of rank-1 tensors
            # The first effective_rank left singular vectors span the column space
            # We need to express each original tensor as a combination of the basis
            # Then we can find new U, V, W factors
            
            # For now, just verify and return the original if we can't easily reconstruct
            # A proper reconstruction would require finding the actual factorization
            
            # Try a simple approach: use the effective_rank left singular vectors
            # as the new "combined" tensors, but we need rank-1 decompositions
            
            # Actually, let's try a different approach: find the null space and 
            # eliminate dependent terms
            
            # For safety, if we can't easily reconstruct, return the original
            pass
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > time_limit - 10:
        return {"U": U, "V": V, "W": W}
    
    # Try iterative rank reduction by checking if we can zero out terms
    # and still maintain the tensor
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try removing each term one at a time and checking if the tensor is preserved
    # This is O(rank * verification_cost) which may be too slow for 56 terms
    
    # Alternative: use the SVD result to find a minimal spanning set
    try:
        U_svd, s, Vt = np.linalg.svd(tensor_stack, full_matrices=False)
        tol = 1e-10
        effective_rank = int(np.sum(s > tol))
        
        if effective_rank < rank:
            # Find a basis of effective_rank columns from tensor_stack
            # Use QR with column pivoting
            Q, R, piv = np.linalg.qr(tensor_stack, mode='complete')
            # The piv gives the column ordering such that R[:effective_rank, :effective_rank] is well-conditioned
            # Actually, let's use a simpler approach: greedy selection
            
            # Select the first effective_rank columns that are linearly independent
            selected = []
            for col_idx in range(rank):
                if len(selected) >= effective_rank:
                    break
                # Check if this column is independent of the selected ones
                if not selected:
                    selected.append(col_idx)
                else:
                    # Form matrix from selected columns and check if adding this column increases rank
                    A = tensor_stack[selected + [col_idx], :].T
                    U_q, s_q, _ = np.linalg.svd(A, full_matrices=False)
                    if len(s_q) > 0 and s_q[0] > tol:
                        # Check if the new column adds rank
                        A_without = tensor_stack[selected, :].T
                        U_wo, s_wo, _ = np.linalg.svd(A_without, full_matrices=False)
                        rank_wo = int(np.sum(s_wo > tol)) if len(s_wo) > 0 else 0
                        if int(np.sum(s_q > tol)) > rank_wo:
                            selected.append(col_idx)
            
            if len(selected) == effective_rank and effective_rank < best_rank:
                # We have a subset of terms that spans the same tensor space
                # But we need to verify that the sum is preserved, not just the span
                # The sum of all 56 tensors must equal the sum of the selected tensors
                # with appropriate coefficients
                
                # This is more complex - we need to find coefficients c_r such that
                # sum_r c_r * tensor_r = original_sum
                # And we want to use only the selected terms
                
                # For now, skip this and just return the original
                pass
    except Exception:
        pass
    
    # Final time check
    if time.time() - start_time > time_limit - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
