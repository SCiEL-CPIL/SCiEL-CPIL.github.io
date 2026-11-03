"""
Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.

Strategy:
1. Start with schoolbook decomposition (rank 144).
2. Use a local search approach to try to reduce the rank by:
   a. Removing redundant terms
   b. Trying to merge/replace pairs of terms with a single term
3. Use time budget to search for better decompositions.

The key insight is that we can use the structure of matrix multiplication
to find lower-rank decompositions. For rectangular matrices, there are
known bounds and algorithms.

For 4x4 * 4x9:
- Schoolbook: rank = 144
- We can potentially do better using block structure

Let's try a computational approach:
1. Start with schoolbook
2. Try to find linear dependencies among the terms
3. Use rank reduction techniques
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Initialize with schoolbook decomposition
    rank = n * m * p  # 144
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that the decomposition is exact."""
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :]))
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # First, verify the schoolbook decomposition
    if not verify(U, V, W):
        # Should never happen, but just in case
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by removing redundant terms
    def try_remove_term(U_mat, V_mat, W_mat, col_idx):
        """Try removing a term and see if the decomposition still works."""
        r = U_mat.shape[1]
        if r <= 1:
            return False, U_mat, V_mat, W_mat
        
        # Create copies without the column
        mask = [i for i in range(r) if i != col_idx]
        U_test = U_mat[:, mask]
        V_test = V_mat[:, mask]
        W_test = W_mat[:, mask]
        
        if verify(U_test, V_test, W_test):
            return True, U_test, V_test, W_test
        return False, U_mat, V_mat, W_mat
    
    # Try removing terms one at a time
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for c in range(U.shape[1]):
            if time.time() > deadline - 30:
                break
            success, U_new, V_new, W_new = try_remove_term(U, V, W, c)
            if success:
                U, V, W = U_new, V_new, W_new
                improved = True
                break
    
    # Try a more sophisticated approach: use linear algebra to find
    # if any column can be expressed as a linear combination of others
    # This is essentially finding the rank of the tensor
    if time.time() < deadline - 20:
        # Try to find a lower rank decomposition using SVD-like approach
        # on the tensor structure
        
        # For each (i,j,k) triple, we need:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is a system of equations. Let's try to find a sparse solution.
        
        # Approach: randomly initialize and use iterative refinement
        def random_decomposition(target_rank):
            """Try to find a decomposition with the given rank."""
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Scale to make the products approximately 1
            # For each (i,j,k), we want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] ≈ 1
            # This is hard to do directly, so let's use a different approach
            
            # Use the fact that we can express the multiplication tensor
            # as a sum of rank-1 tensors
            pass
        
        # Alternative: use known results for rectangular matrix multiplication
        # For 4x4 * 4x9, we can potentially use a block decomposition
        
        # Let's try: split the 4x9 matrix into three 4x3 matrices
        # C = A * [B1 | B2 | B3] where B1, B2, B3 are 4x3
        # C = [A*B1 | A*B2 | A*B3]
        
        # For each A*Bi, we can use a 4x4 * 4x3 decomposition
        # The rank for 4x4 * 4x3 is at most 4*4*3 = 48
        
        # So total rank would be at most 3 * 48 = 144 (same as schoolbook)
        
        # But we might be able to do better by sharing structure between the blocks
        
        # Let's try a different approach: use the fact that 9 = 4 + 4 + 1
        # C = A * [B1 | B2 | B3] where B1, B2 are 4x4 and B3 is 4x1
        # C = [A*B1 | A*B2 | A*B3]
        
        # For A*B1 and A*B2, we can use 4x4 * 4x4 decompositions
        # For A*B3, we can use 4x4 * 4x1 decompositions
        
        # The rank for 4x4 * 4x4 is known to be at most 49 (Lader's algorithm)
        # The rank for 4x4 * 4x1 is at most 4*4*1 = 16
        
        # So total rank would be at most 49 + 49 + 16 = 114
        
        # Let's try to construct this decomposition
        if time.time() < deadline - 15:
            # Use a known 4x4 * 4x4 decomposition of rank 49
            # This is complex, so let's try a simpler approach first
            
            # Try to find a decomposition using iterative methods
            # Start with schoolbook and try to reduce rank using gradient descent
            # on the entries of U, V, W
            
            # Define the objective: minimize sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            # subject to the constraint that the solution is exact
            
            # This is a non-convex optimization problem, but we can try local search
            
            # For now, let's just try a few random perturbations and see if we can
            # find a lower rank solution
            
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
            
            for attempt in range(10):
                if time.time() > deadline - 15:
                    break
                
                # Try a random perturbation of the current solution
                U_pert = U + 0.1 * np.random.randn(*U.shape)
                V_pert = V + 0.1 * np.random.randn(*V.shape)
                W_pert = W + 0.1 * np.random.randn(*W.shape)
                
                # Check if the perturbed solution is still valid
                if verify(U_pert, V_pert, W_pert, tol=1e-8):
                    # Try to reduce rank by removing columns
                    for c in range(U_pert.shape[1]):
                        if time.time() > deadline - 15:
                            break
                        success, U_new, V_new, W_new = try_remove_term(U_pert, V_pert, W_pert, c)
                        if success and U_new.shape[1] < best_rank:
                            best_U, best_V, best_W = U_new, V_new, W_new
                            best_rank = U_new.shape[1]
                            U, V, W = U_new, V_new, W_new
                            break
    
    # Convert to lists and return
    U_final = U.tolist()
    V_final = V.tolist()
    W_final = W.tolist()
    
    # Ensure all entries are integers or half-integers (as required)
    # Round to nearest 0.5
    def round_to_half(x):
        return round(x * 2) / 2.0
    
    U_final = [[round_to_half(val) for val in row] for row in U_final]
    V_final = [[round_to_half(val) for val in row] for row in V_final]
    W_final = [[round_to_half(val) for val in row] for row in W_final]
    
    # Final verification
    U_check = np.array(U_final)
    V_check = np.array(V_final)
    W_check = np.array(W_final)
    
    if not verify(U_check, V_check, W_check, tol=1e-8):
        # If rounding broke the solution, use the unrounded version
        U_final = U.tolist()
        V_final = V.tolist()
        W_final = W.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
