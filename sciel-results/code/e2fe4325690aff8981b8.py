"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96

    # Build schoolbook solution
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank

    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]
        
        # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
        
        # T should be all ones
        return np.allclose(T_recon, 1.0, atol=tol)

    def try_rank_reduction(current_U, current_V, current_W, current_rank, max_removals=8):
        """Try to reduce rank by removing terms and solving for adjustments."""
        U_arr = np.array(current_U, dtype=float)
        V_arr = np.array(current_V, dtype=float)
        W_arr = np.array(current_W, dtype=float)
        
        best_U, best_V, best_W = current_U, current_V, current_W
        best_r = current_rank
        
        for attempt in range(max_removals):
            if time.time() > deadline - 5.0:
                break
            
            # Try removing one column at a time
            removed = False
            for c_remove in range(best_r):
                if time.time() > deadline - 5.0:
                    break
                
                # Remove column c_remove
                mask = [c for c in range(best_r) if c != c_remove]
                if not mask:
                    break
                
                U_reduced = U_arr[:, mask]
                V_reduced = V_arr[:, mask]
                W_reduced = W_arr[:, mask]
                
                # Check if this still works (it won't, since we're removing a term)
                # Instead, we need to see if we can adjust the remaining terms
                # to compensate. This is a linear system problem.
                
                # For now, just check if removing this column and scaling others works
                # This is a simplified approach
                
                removed = True
                break
            
            if not removed:
                break
        
        return best_U, best_V, best_W, best_r

    # Strategy: Use the known structure of matrix multiplication
    # For 3x4x8, we can potentially exploit the rectangular nature
    
    # Known result: The rank of 3x4x8 matrix multiplication is at most 96 (schoolbook)
    # Let's try to find a lower rank by exploiting symmetries
    
    # Approach: Try to merge terms that share indices
    # For example, terms with the same i can potentially be combined
    
    # Let's try a more systematic approach:
    # 1. Start with schoolbook
    # 2. Try to find linear dependencies among the terms
    
    # Build the tensor as a vector
    # T[i,j,k] = 1 for all i,j,k
    # Each term c contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's try to see if we can express some terms as linear combinations
    # of others by solving a linear system
    
    # For a given set of terms to keep, check if the tensor can be reconstructed
    # This is equivalent to checking if the remaining terms span the same space
    
    # Simpler approach: Try specific rank values and see if a solution exists
    
    # For 3x4x8, let's try ranks from 95 down to 80
    # This is a search, so we'll do it greedily
    
    # Actually, let's use a different strategy:
    # The matrix multiplication tensor has a specific structure that allows
    # for rank reduction when dimensions are not all equal.
    
    # For 3x4x8, one approach is to use the fact that 3*8=24 and 4*8=32,
    # and there might be ways to share computation.
    
    # Let's try to find a rank-90 decomposition by hand-crafting
    # based on known algorithms for rectangular matrix multiplication.
    
    # Known: For multiplying (m x n) by (n x p), the rank is at most m*n*p
    # but can be lower. For 3x4x8, let's see if we can do better than 96.
    
    # One known technique is to use the "Winograd" style of algorithms
    # which can reduce the rank for certain dimensions.
    
    # For now, let's return the schoolbook solution if we can't find
    # a better one quickly.
    
    if time.time() > deadline - 20.0:
        return best_result
    
    # Try to find a lower rank by searching
    # We'll use a greedy approach: try to remove one term at a time
    # and see if we can adjust the others to compensate.
    
    # This is complex, so let's try a simpler heuristic:
    # Look for terms that can be merged based on shared indices.
    
    # For example, if two terms have the same i, j, or k, they might
    # be mergeable.
    
    # Let's try to find a rank-90 solution by removing 6 terms
    # and adjusting the remaining 90 terms.
    
    # This requires solving a linear system, which we can do with numpy.
    
    # For now, return the schoolbook solution
    return best_result
# EVOLVE-BLOCK-END
