"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for 150s limit
    
    # Schoolbook decomposition (rank 40)
    target_rank = n * m * p  # 40
    
    U_school = np.zeros((n * m, target_rank))
    V_school = np.zeros((m * p, target_rank))
    W_school = np.zeros((p * n, target_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U_school[idx_u, col] = 1.0
                V_school[idx_v, col] = 1.0
                W_school[idx_w, col] = 1.0
                col += 1
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Compute the reconstructed tensor
        # T[idx_u, idx_v, idx_w] = sum_c U[idx_u,c] * V[idx_v,c] * W[idx_w,c]
        # For efficiency, compute via matrix operations
        # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # Reshape: for each c, outer product
        
        # Check all 40 one-positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        
        # Check that zero positions are zero (sample some)
        # Full check would be expensive, but for small dimensions it's fine
        for idx_u in range(n * m):
            for idx_v in range(m * p):
                for idx_w in range(p * n):
                    # Check if this is a one-position
                    is_one = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if (i * m + j == idx_u and j * p + k == idx_v and k * n + i == idx_w):
                                    is_one = True
                                    break
                            if is_one:
                                break
                        if is_one:
                            break
                    if not is_one:
                        val = 0.0
                        for c in range(r):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        if abs(val) > 1e-10:
                            return False
        return True
    
    # First, verify schoolbook
    if not verify(U_school, V_school, W_school):
        # Should never happen
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = target_rank
    
    # Try to find a lower rank decomposition
    # Strategy: Use the fact that for 2x4x5, we can try rank 39, 38, etc.
    # Use a greedy approach: try to remove one term at a time and see if we can adjust
    # the remaining terms to compensate.
    
    # Alternative approach: Use the Strassen-like ideas or known results
    # For 2x4x5, the tensor rank might be less than 40.
    
    # Let's try a more systematic approach:
    # For rank r < 40, we need to find U (8xr), V (20xr), W (10xr)
    # such that the tensor equation is satisfied.
    
    # This is a nonlinear system. Let's try to use the null space approach.
    
    # Build the full tensor as a vector
    tensor_size = n * m * m * p * p * n  # 8 * 20 * 10 = 1600
    T_vec = np.zeros(tensor_size)
    
    # Fill in the one-positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                flat_idx = idx_u * (m * p) * (p * n) + idx_v * (p * n) + idx_w
                T_vec[flat_idx] = 1.0
    
    # For each rank-1 term, we can represent it as a vector
    # Let's try to find a rank 39 decomposition by perturbing the schoolbook
    
    # Approach: For each column c to remove, try to find coefficients
    # for the remaining columns that satisfy the equation.
    # This is equivalent to solving a system of linear equations if we fix
    # two of the three matrices.
    
    # Let's try: fix V and W, solve for U. Or fix U and W, solve for V.
    
    # Actually, let's try a different approach:
    # Use the fact that the tensor is sparse and try to find a sparse decomposition.
    
    # For now, let's try a random search for rank 39
    if time.time() < deadline:
        # Try random rank 39 decompositions
        np.random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            r = 39
            # Initialize with schoolbook minus one random column, then perturb
            U_try = np.zeros((n * m, r))
            V_try = np.zeros((m * p, r))
            W_try = np.zeros((p * n, r))
            
            # Copy first 39 columns of schoolbook
            skip_col = attempt % target_rank
            col_idx = 0
            for c in range(target_rank):
                if c == skip_col:
                    continue
                U_try[:, col_idx] = U_school[:, c]
                V_try[:, col_idx] = V_school[:, c]
                W_try[:, col_idx] = W_school[:, c]
                col_idx += 1
            
            # Add small random perturbations
            U_try += np.random.randn(n * m, r) * 0.01
            V_try += np.random.randn(m * p, r) * 0.01
            W_try += np.random.randn(p * n, r) * 0.01
            
            if verify(U_try, V_try, W_try):
                best_U = U_try
                best_V = V_try
                best_W = W_try
                best_rank = r
                break
    
    # Try rank 38
    if best_rank == 40 and time.time() < deadline:
        np.random.seed(123)
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            r = 38
            U_try = np.zeros((n * m, r))
            V_try = np.zeros((m * p, r))
            W_try = np.zeros((p * n, r))
            
            # Use first 38 columns
            for c in range(r):
                U_try[:, c] = U_school[:, c]
                V_try[:, c] = V_school[:, c]
                W_try[:, c] = W_school[:, c]
            
            U_try += np.random.randn(n * m, r) * 0.01
            V_try += np.random.randn(m * p, r) * 0.01
            W_try += np.random.randn(p * n, r) * 0.01
            
            if verify(U_try, V_try, W_try):
                best_U = U_try
                best_V = V_try
                best_W = W_try
                best_rank = r
                break
    
    # If we still have rank 40, try a more systematic approach
    if best_rank == 40 and time.time() < deadline:
        # Try to use the linear algebra approach
        # The key insight: if we can find a non-trivial linear combination
        # of the rank-1 tensors that sums to zero, we might be able to
        # restructure the decomposition.
        
        # Build the matrix of rank-1 tensors
        tensors = []
        for c in range(target_rank):
            u = U_school[:, c]
            v = V_school[:, c]
            w = W_school[:, c]
            vec = np.outer(np.outer(u, v), w).flatten()
            tensors.append(vec)
        
        tensor_matrix = np.column_stack(tensors)  # shape (1600, 40)
        
        # Check if the columns are linearly independent
        try:
            rank = np.linalg.matrix_rank(tensor_matrix, tol=1e-10)
        except:
            rank = target_rank
        
        if rank < target_rank:
            # There's a linear dependency
            # Find the null space
            try:
                _, s, vh = np.linalg.svd(tensor_matrix, full_matrices=False)
                null_dim = target_rank - rank
                null_vecs = vh[rank:]  # shape (null_dim, 40)
                
                # For each null vector, we have a relationship
                # sum_c null_vec[c] * tensor_c = 0
                # This means we can modify the decomposition
            except:
                pass
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U = U_school.copy()
        best_V = V_school.copy()
        best_W = W_school.copy()
        best_rank = target_rank
    
    # Round to nearest half-integer if possible
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    best_U = round_to_half(best_U)
    best_V = round_to_half(best_V)
    best_W = round_to_half(best_W)
    
    # Final check after rounding
    if not verify(best_U, best_V, best_W):
        # Use unrounded values
        best_U = best_U.copy()
        best_V = best_V.copy()
        best_W = best_W.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
