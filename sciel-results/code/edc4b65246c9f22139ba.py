import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target ranks to try, in increasing order
    target_ranks = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 48, 50, 52, 54]
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute matricizations of T for ALS
    # T_U: shape (n*m, (m*p)*(p*n)) - flatten V,W indices
    # T_V: shape (m*p, (n*m)*(p*n)) - flatten U,W indices  
    # T_W: shape (p*n, (n*m)*(m*p)) - flatten U,V indices
    
    # For ALS, we need to solve:
    # U = argmin ||T - sum_c U_c * V_c * W_c||
    # This is equivalent to solving linear systems.
    
    # Let's use the fact that for fixed V and W, we can solve for U via:
    # T = U * (V^T ⊗ W^T)  where ⊗ is Kronecker product
    # Actually, let's think more carefully.
    
    # T is a 3-way tensor. We want T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # This means T[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
    
    # For fixed V and W, we can vectorize:
    # Let T_vec be T reshaped to (n*m, (m*p)*(p*n))
    # Then T_vec = U * (V ⊗ W)^T  where V⊗W is the Kronecker product
    # Wait, let me be more careful.
    
    # T[a, (b,c)] = sum_k U[a,k] * V[b,k] * W[c,k]
    # Let M_VW be a matrix of shape ((m*p)*(p*n), rank) where
    # M_VW[(b,c), k] = V[b,k] * W[c,k]
    # Then T_vec = U * M_VW^T
    # So U = T_vec * (M_VW^T)^+ = T_vec * pinv(M_VW^T)
    
    # Similarly for V and W.
    
    def solve_factor(T_mat, other1, other2):
        """
        Solve for factor given target matricization and two other factors.
        T_mat: shape (dim, dim2*dim3) 
        other1: shape (dim2, rank)
        other2: shape (dim3, rank)
        Returns: shape (dim, rank)
        """
        # Build M = other1 ⊗ other2 element-wise per column
        # M[(b,c), k] = other1[b,k] * other2[c,k]
        # We need to reshape other1 and other2 appropriately
        
        dim2 = other1.shape[0]
        dim3 = other2.shape[0]
        rank = other1.shape[1]
        
        # M has shape (dim2*dim3, rank)
        # M[(b,c), k] = other1[b,k] * other2[c,k]
        
        # Reshape other1 to (dim2, 1, rank) and other2 to (1, dim3, rank)
        # Then multiply and reshape
        M = (other1.reshape(dim2, 1, rank) * other2.reshape(1, dim3, rank)).reshape(dim2 * dim3, rank)
        
        # Now T_mat = factor * M^T
        # factor = T_mat * pinv(M^T)
        # Use least squares: factor = lstsq(M^T, T_mat^T)^T
        # Or: factor = T_mat @ np.linalg.pinv(M.T)
        
        # More efficient: solve M^T @ factor^T = T_mat^T
        # factor^T = lstsq(M^T, T_mat^T)
        
        try:
            factor_T, residuals, rank_2, sv = np.linalg.lstsq(M.T, T_mat.T, rcond=None)
            return factor_T.T
        except:
            # Fallback to pinv
            return T_mat @ np.linalg.pinv(M.T)
    
    # Precompute matricizations
    T_U = T.reshape(n * m, m * p * p * n)  # (n*m, (m*p)*(p*n))
    T_V = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)  # (m*p, (n*m)*(p*n))
    T_W = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)  # (p*n, (n*m)*(m*p))
    
    # ALS solver
    def als_solve(T_U, T_V, T_W, U, V, W, max_iter=200, tol=1e-10):
        """
        Alternating least squares for tensor decomposition.
        """
        for it in range(max_iter):
            # Solve for U given V, W
            U_new = solve_factor(T_U, V, W)
            # Solve for V given U, W
            V_new = solve_factor(T_V, U_new, W)
            # Solve for W given U, V
            W_new = solve_factor(T_W, U_new, V_new)
            
            # Check convergence
            delta_U = np.linalg.norm(U_new - U)
            delta_V = np.linalg.norm(V_new - V)
            delta_W = np.linalg.norm(W_new - W)
            
            U, V, W = U_new, V_new, W_new
            
            if delta_U + delta_V + delta_W < tol:
                break
        
        return U, V, W
    
    def check_decomposition(U, V, W, T):
        """
        Check if sum_c U[:,c] * V[:,c] * W[:,c] = T
        """
        # Compute reconstructed tensor
        T_rec = np.zeros_like(T)
        for c in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        # Check exact match (within tolerance for floating point)
        max_err = np.max(np.abs(T_rec - T))
        return max_err, max_err < 1e-8
    
    # Try each target rank
    for r in target_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        n_attempts = 50 if r <= 38 else 30 if r <= 45 else 20
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(hash((r, attempt)) % (2**32))
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Run ALS
            U, V, W = als_solve(T_U, T_V, T_W, U, V, W, max_iter=100)
            
            # Check if we found a valid decomposition
            max_err, valid = check_decomposition(U, V, W, T)
            
            if valid:
                # Found a valid decomposition of rank r
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
        
        if found:
            # If we found rank r, we can stop if it's minimal
            # But we should check if lower ranks are possible
            # For now, break since we're trying in increasing order
            break
    
    # If no improvement found, return schoolbook
    if best_result is None:
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    else:
        return best_result
