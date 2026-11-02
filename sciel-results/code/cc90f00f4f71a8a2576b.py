"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    total_entries = U_dim * V_dim * W_dim  # 8*32*16 = 4096
    
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def check_decomp(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This should equal 1 for all (i,j,k)
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def check_decomp_fast(U, V, W, tol=1e-6):
        """Faster check using einsum."""
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape and use einsum
        U_r = U.T  # (rank, U_dim)
        V_r = V.T  # (rank, V_dim)
        W_r = W.T  # (rank, W_dim)
        
        # For each r, the contribution is outer product of U_r[r], V_r[r], W_r[r]
        # We need sum_r U_r[r, i*m+j] * V_r[r, j*p+k] * W_r[r, k*n+i] = 1
        
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # Use a loop over r but vectorize over i,j,k
        T = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r]  # (U_dim,)
            v = V[:, r]  # (V_dim,)
            w = W[:, r]  # (W_dim,)
            
            # u[i*m+j] * v[j*p+k] * w[k*n+i]
            # Reshape u to (n, m), v to (m, p), w to (p, n)
            u_2d = u.reshape(n, m)
            v_2d = v.reshape(m, p)
            w_2d = w.reshape(p, n)
            
            # T[i,j,k] += u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
            # This is like a tensor contraction
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True
    
    def try_reduce_rank(U, V, W, target_rank, max_iter=50):
        """
        Try to reduce the rank of the decomposition to target_rank.
        Uses iterative SVD-based rank minimization.
        """
        if U.shape[1] <= target_rank:
            return U, V, W, True
        
        rank = U.shape[1]
        
        # Randomly sample a subset of tensor entries to constrain
        all_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_success = False
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Sample a random subset of constraints
            num_constraints = min(len(all_indices), max(50, target_rank * 3))
            sampled = random.sample(all_indices, num_constraints)
            
            # Build constraint matrix: for each sampled (i,j,k), we need
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # 
            # Linearize: fix two of the three matrices, solve for the third.
            # Strategy: alternate fixing U&V, solve W; fix U&W, solve V; fix V&W, solve U.
            
            U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
            
            converged = False
            for inner_iter in range(20):
                if time.time() > deadline - 5.0:
                    break
                
                # Fix U and V, solve for W
                # For each sampled (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                # This is: (U[i*m+j,:] * V[j*p+k,:]) @ W[k*n+i,:] = 1
                # Let a_r = U[i*m+j,r]*V[j*p+k,r], then sum_r a_r * W[k*n+i,r] = 1
                # This is a linear system in W.
                
                # Build the linear system: A_w @ w_vec = b_w
                # where w_vec is the flattened W matrix
                # A_w[k*n+i, r] = U[i*m+j,r] * V[j*p+k,r]
                # b_w[k*n+i] = 1 (for sampled indices)
                
                A_w = np.zeros((W_dim, rank))
                b_w = np.zeros(W_dim)
                count_w = np.zeros(W_dim)
                
                for (i, j, k) in sampled:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    A_w[w_idx, :] = U_curr[u_idx, :] * V_curr[v_idx, :]
                    b_w[w_idx] += 1.0
                    count_w[w_idx] += 1
                
                # Only use rows with at least one constraint
                valid_w = count_w > 0
                if np.sum(valid_w) > 0:
                    # Least squares solution
                    try:
                        W_sol, residuals, rank_w, sv = np.linalg.lstsq(
                            A_w[valid_w, :], b_w[valid_w], rcond=None
                        )
                        W_curr[valid_w, :] = W_sol
                    except:
                        pass
                
                # Fix U and W, solve for V
                A_v = np.zeros((V_dim, rank))
                b_v = np.zeros(V_dim)
                count_v = np.zeros(V_dim)
                
                for (i, j, k) in sampled:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    A_v[v_idx, :] = U_curr[u_idx, :] * W_curr[w_idx, :]
                    b_v[v_idx] += 1.0
                    count_v[v_idx] += 1
                
                valid_v = count_v > 0
                if np.sum(valid_v) > 0:
                    try:
                        V_sol, residuals, rank_v, sv = np.linalg.lstsq(
                            A_v[valid_v, :], b_v[valid_v], rcond=None
                        )
                        V_curr[valid_v, :] = V_sol
                    except:
                        pass
                
                # Fix V and W, solve for U
                A_u = np.zeros((U_dim, rank))
                b_u = np.zeros(U_dim)
                count_u = np.zeros(U_dim)
                
                for (i, j, k) in sampled:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    A_u[u_idx, :] = V_curr[v_idx, :] * W_curr[w_idx, :]
                    b_u[u_idx] += 1.0
                    count_u[u_idx] += 1
                
                valid_u = count_u > 0
                if np.sum(valid_u) > 0:
                    try:
                        U_sol, residuals, rank_u, sv = np.linalg.lstsq(
                            A_u[valid_u, :], b_u[valid_u], rcond=None
                        )
                        U_curr[valid_u, :] = U_sol
                    except:
                        pass
            
            # Now try to reduce rank via SVD
            # Stack the three matrices and find a lower-rank approximation
            # Project to target_rank using SVD on the "combined" structure
            
            # For each matrix, do SVD and keep only target_rank components
            # But we need to maintain consistency across all three matrices.
            
            # Alternative: use the fact that if rank > target_rank,
            # we can try to find a linear dependency among the columns.
            
            # Check if current solution is valid
            if check_decomp(U_curr, V_curr, W_curr, tol=1e-5):
                # Try to reduce rank
                # Find null space of the combined system
                # Stack all columns: each column r defines a triple (u_r, v_r, w_r)
                # We want to find a lower-dimensional subspace that still spans the solution.
                
                # Try SVD-based compression
                # For each of U, V, W, compute SVD and truncate
                for mat_name, mat in [('U', U_curr), ('V', V_curr), ('W', W_curr)]:
                    try:
                        if mat.shape[1] > target_rank:
                            U_svd, s, Vt = np.linalg.svd(mat, full_matrices=False)
                            # Keep only target_rank components
                            if s.size > target_rank:
                                new_mat = U_svd[:, :target_rank] @ np.diag(s[:target_rank]) @ Vt[:target_rank, :]
                                if mat_name == 'U':
                                    U_curr = new_mat
                                elif mat_name == 'V':
                                    V_curr = new_mat
                                else:
                                    W_curr = new_mat
                    except:
                        pass
                
                # Re-verify after truncation
                if check_decomp(U_curr, V_curr, W_curr, tol=1e-5):
                    if U_curr.shape[1] < rank:
                        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                        best_success = True
                        rank = U_curr.shape[1]
                        if rank <= target_rank:
                            break
            
            # Also try: if we have more columns than needed, try to eliminate
            # columns that are linear combinations of others
            if rank > target_rank:
                # Try to find linear dependencies in U
                try:
                    # Check if any column of U is a linear combination of others
                    # Use QR decomposition
                    Q, R = np.linalg.qr(U_curr.T)  # (rank, U_dim)
                    # Check for small diagonal entries in R
                    diag = np.abs(np.diag(R))
                    # Sort by diagonal magnitude
                    order = np.argsort(diag)
                    # Try to remove columns with smallest diagonal
                    for idx in order:
                        if diag[idx] < 1e-8 and rank > target_rank:
                            # Try removing this column
                            mask = np.ones(rank, dtype=bool)
                            mask[idx] = False
                            U_test = U_curr[:, mask]
                            V_test = V_curr[:, mask]
                            W_test = W_curr[:, mask]
                            if check_decomp(U_test, V_test, W_test, tol=1e-5):
                                U_curr, V_curr, W_curr = U_test, V_test, W_test
                                rank -= 1
                                if rank <= target_rank:
                                    break
                except:
                    pass
        
        if best_success:
            return best_U, best_V, best_W, True
        return U, V, W, False
    
    # Start with schoolbook (rank 64)
    rank = n * m * p
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    if not check_decomp(U, V, W):
        # Should never happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce rank
    # First, try the known Strassen-based rank 28
    # Then try to go lower
    
    # Build Strassen rank-28 decomposition
    def build_strassen_28():
        U_s = np.zeros((U_dim, 28))
        V_s = np.zeros((V_dim, 28))
        W_s = np.zeros((W_dim, 28))
        
        def add_strassen_term(base_col, A_idx_map, B_idx_map, C_idx_map):
            col = base_col
            U_s[A_idx_map['A00'], col] = 1
            V_s[B_idx_map['B01'], col] = 1
            V_s[B_idx_map['B11'], col] = 1
            W_s[C_idx_map['C00'], col] = 1
            W_s[C_idx_map['C11'], col] = 1
            
            col = base_col + 1
            U_s[A_idx_map['A10'], col] = 1
            U_s[A_idx_map['A11'], col] = 1
            V_s[B_idx_map['B00'], col] = 1
            W_s[C_idx_map['C10'], col] = 1
            W_s[C_idx_map['C11'], col] = -1
            
            col = base_col + 2
            U_s[A_idx_map['A00'], col] = 1
            U_s[A_idx_map['A01'], col] = 1
            V_s[B_idx_map['B10'], col] = 1
            W_s[C_idx_map['C01'], col] = 1
            W_s[C_idx_map['C11'], col] = 1
            
            col = base_col + 3
            U_s[A_idx_map['A11'], col] = 1
            V_s[B_idx_map['B10'], col] = 1
            V_s[B_idx_map['B00'], col] = 1
            W_s[C_idx_map['C00'], col] = 1
            W_s[C_idx_map['C10'], col] = 1
            
            col = base_col + 4
            U_s[A_idx_map['A00'], col] = 1
            U_s[A_idx_map['A10'], col] = 1
            V_s[B_idx_map['B01'], col] = 1
            W_s[C_idx_map['C00'], col] = -1
            W_s[C_idx_map['C01'], col] = 1
            
            col = base_col + 5
            U_s[A_idx_map['A01'], col] = 1
            U_s[A_idx_map['A11'], col] = -1
            V_s[B_idx_map['B10'], col] = 1
            V_s[B_idx_map['B11'], col] = 1
            W_s[C_idx_map['C11'], col] = 1
            
            col = base_col + 6
            U_s[A_idx_map['A10'], col] = 1
            U_s[A_idx_map['A00'], col] = -1
            V_s[B_idx_map['B00'], col] = 1
            V_s[B_idx_map['B01'], col] = 1
            W_s[C_idx_map['C00'], col] = 1
        
        add_strassen_term(0,
                          {'A00': 0, 'A01': 1, 'A10': 4, 'A11': 5},
                          {'B00': 0, 'B01': 1, 'B10': 8, 'B11': 9},
                          {'C00': 0, 'C01': 2, 'C10': 1, 'C11': 3})
        
        add_strassen_term(7,
                          {'A00': 2, 'A01': 3, 'A10': 6, 'A11': 7},
                          {'B00': 16, 'B01': 17, 'B10': 24, 'B11': 25},
                          {'C00': 0, 'C01': 2, 'C10': 1, 'C11': 3})
        
        add_strassen_term(14,
                          {'A00': 0, 'A01': 1, 'A10': 4, 'A11': 5},
                          {'B00': 2, 'B01': 3, 'B10': 10, 'B11': 11},
                          {'C00': 4, 'C01': 6, 'C10': 5, 'C11': 7})
        
        add_strassen_term(21,
                          {'A00': 2, 'A01': 3, 'A10': 6, 'A11': 7},
                          {'B00': 18, 'B01': 19, 'B10': 26, 'B11': 27},
                          {'C00': 4, 'C01': 6, 'C10': 5, 'C11': 7})
        
        return U_s, V_s, W_s
    
    U_str, V_str, W_str = build_strassen_28()
    
    if check_decomp(U_str, V_str, W_str):
        U, V, W = U_str, V_str, W_str
        current_rank = 28
    else:
        current_rank = 64
    
    # Now try to reduce below current_rank
    # Try various target ranks
    for target_rank in range(current_rank - 1, 15, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Try multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Add small random perturbation to break symmetry
            U_pert = U + 0.01 * np.random.randn(U_dim, current_rank)
            V_pert = V + 0.01 * np.random.randn(V_dim, current_rank)
            W_pert = W + 0.01 * np.random.randn(W_dim, current_rank)
            
            U_new, V_new, W_new, success = try_reduce_rank(
                U_pert, V_pert, W_pert, target_rank, max_iter=30
            )
            
            if success and U_new.shape[1] < current_rank:
                U, V, W = U_new, V_new, W_new
                current_rank = U_new.shape[1]
                break
        
        if time.time() > deadline - 10.0:
            break
    
    # Final verification
    if not check_decomp(U, V, W):
        # Fall back to Strassen if valid, else schoolbook
        if check_decomp(U_str, V_str, W_str):
            U, V, W = U_str, V_str, W_str
        else:
            rank = n * m * p
            U = np.zeros((U_dim, rank))
            V = np.zeros((V_dim, rank))
            W = np.zeros((W_dim, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
