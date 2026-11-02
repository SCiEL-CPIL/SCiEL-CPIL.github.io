import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _build_true_tensor(n, m, p):
    """Build the true tensor for matrix multiplication."""
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0
    return T

def _build_rank32():
    """Build the known rank-32 solution using Strassen for 2x2x2 blocks."""
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # Standard Strassen for 2x2x2
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0*p+k, 1*p+k
        # W indices: k*n+0, k*n+1
        
        # M1 = (A00+A11)(B00+B11)
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    # 9 columns in B.
    # Split into 4 blocks of 2 columns and 1 block of 1 column.
    # Columns: 0,1 | 2,3 | 4,5 | 6,7 | 8
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, T_true):
    """Compute the residual tensor and its norm."""
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    residual = M - T_true
    return residual, np.linalg.norm(residual)

def _try_reduce_rank(U, V, W, T_true, target_rank, time_budget):
    """
    Try to reduce the rank of the decomposition to target_rank.
    Uses a smooth epigraph formulation with multiple column removal strategies.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W, True

    cols_to_remove = current_rank - target_rank
    start_time = time.time()
    
    # Strategy: Try multiple approaches to remove columns
    # Approach 1: Remove columns with smallest contribution to tensor
    # Approach 2: Use epigraph formulation with auxiliary variable t
    
    best_result = None
    best_norm = float('inf')
    
    # Try different column removal strategies
    for strategy in range(3):
        if time.time() - start_time > time_budget:
            break
        
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        current_rank_curr = current_rank
        removed = 0
        
        for i in range(cols_to_remove):
            if time.time() - start_time > time_budget:
                break
                
            # Choose column to remove based on strategy
            if strategy == 0:
                # Smallest combined norm
                norms = [np.linalg.norm(U_curr[:, c]) + np.linalg.norm(V_curr[:, c]) + np.linalg.norm(W_curr[:, c]) for c in range(current_rank_curr)]
                target_col = np.argmin(norms)
            elif strategy == 1:
                # Column with smallest contribution to tensor (using first-order approximation)
                contributions = []
                for c in range(current_rank_curr):
                    # Approximate contribution as norm of U[:,c] * V[:,c] * W[:,c] outer product
                    contrib = np.linalg.norm(U_curr[:, c]) * np.linalg.norm(V_curr[:, c]) * np.linalg.norm(W_curr[:, c])
                    contributions.append(contrib)
                target_col = np.argmin(contributions)
            else:
                # Random column
                target_col = np.random.randint(0, current_rank_curr)
            
            # Zero out this column
            U_curr[:, target_col] = 0
            V_curr[:, target_col] = 0
            W_curr[:, target_col] = 0
            
            # Optimize using epigraph formulation
            # Minimize ||M - T||^2 + lambda * (||U_target||^2 + ||V_target||^2 + ||W_target||^2)
            # where target column should be near zero
            
            def objective(params, target_col_idx=target_col, curr_rank=current_rank_curr):
                U_flat = params[:nm * curr_rank]
                V_flat = params[nm * curr_rank : nm * curr_rank + mp * curr_rank]
                W_flat = params[nm * curr_rank + mp * curr_rank : nm * curr_rank + mp * curr_rank + pn * curr_rank]
                
                U_mat = U_flat.reshape(nm, curr_rank)
                V_mat = V_flat.reshape(mp, curr_rank)
                W_mat = W_flat.reshape(pn, curr_rank)
                
                M = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
                residual = M - T_true
                base_loss = np.sum(residual**2)
                
                # Penalty for target column being non-zero
                col_penalty = 50.0 * (np.sum(U_mat[:, target_col_idx]**2) + 
                                      np.sum(V_mat[:, target_col_idx]**2) + 
                                      np.sum(W_mat[:, target_col_idx]**2))
                
                return base_loss + col_penalty
            
            x0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
            
            try:
                from scipy.optimize import minimize
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10})
                x_opt = result.x
            except Exception:
                break
                
            U_flat = x_opt[:nm * current_rank_curr]
            V_flat = x_opt[nm * current_rank_curr : nm * current_rank_curr + mp * current_rank_curr]
            W_flat = x_opt[nm * current_rank_curr + mp * current_rank_curr : nm * current_rank_curr + mp * current_rank_curr + pn * current_rank_curr]
            
            U_curr = U_flat.reshape(nm, current_rank_curr)
            V_curr = V_flat.reshape(mp, current_rank_curr)
            W_curr = W_flat.reshape(pn, current_rank_curr)
            
            # Check if column is effectively zero
            col_norm = np.linalg.norm(U_curr[:, target_col]) + np.linalg.norm(V_curr[:, target_col]) + np.linalg.norm(W_curr[:, target_col])
            
            if col_norm < 1e-4:
                U_curr = np.delete(U_curr, target_col, axis=1)
                V_curr = np.delete(V_curr, target_col, axis=1)
                W_curr = np.delete(W_curr, target_col, axis=1)
                current_rank_curr -= 1
                removed += 1
            else:
                break
        
        if removed == cols_to_remove:
            # Final high-precision polish
            def final_objective(params):
                U_flat = params[:nm * target_rank]
                V_flat = params[nm * target_rank : nm * target_rank + mp * target_rank]
                W_flat = params[nm * target_rank + mp * target_rank : nm * target_rank + mp * target_rank + pn * target_rank]
                
                U_mat = U_flat.reshape(nm, target_rank)
                V_mat = V_flat.reshape(mp, target_rank)
                W_mat = W_flat.reshape(pn, target_rank)
                
                M = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
                residual = M - T_true
                return np.sum(residual**2)
            
            x0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
            try:
                from scipy.optimize import minimize
                result = minimize(final_objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-12})
                x_opt = result.x
                U_flat = x_opt[:nm * target_rank]
                V_flat = x_opt[nm * target_rank : nm * target_rank + mp * target_rank]
                W_flat = x_opt[nm * target_rank + mp * target_rank : nm * target_rank + mp * target_rank + pn * target_rank]
                
                U_final = U_flat.reshape(nm, target_rank)
                V_final = V_flat.reshape(mp, target_rank)
                W_final = W_flat.reshape(pn, target_rank)
                
                _, norm = _compute_residual(U_final, V_final, W_final, T_true)
                if norm < 1e-8 and norm < best_norm:
                    best_norm = norm
                    best_result = (U_final, V_final, W_final)
            except Exception:
                pass
    
    if best_result is not None:
        return best_result[0], best_result[1], best_result[2], True
    
    return None, None, None, False

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    T_true = _build_true_tensor(n, m, p)
    
    # Start with rank 32
    U, V, W = _build_rank32()
    
    def to_list(arr):
        return arr.tolist()
    
    # Try to reduce rank
    # We want to try 28, 27, 26...
    # But step-by-step is safer.
    
    current_U, current_V, current_W = U, V, W
    current_rank = 32
    
    # Try to reduce step by step
    target_ranks = [31, 30, 29, 28, 27, 26]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
            
        # Calculate time budget for this attempt
        time_left = deadline - time.time()
        budget = min(25.0, time_left * 0.4) # Don't spend too much on one step
        
        if budget < 8.0:
            break
            
        new_U, new_V, new_W, success = _try_reduce_rank(current_U, current_V, current_W, T_true, target_rank, budget)
        
        if success:
            current_U, current_V, current_W = new_U, new_V, new_W
            current_rank = target_rank
            # Check residual
            _, norm = _compute_residual(current_U, current_V, current_W, T_true)
            if norm > 1e-6:
                # Something went wrong, revert
                current_U, current_V, current_W = U, V, W
                current_rank = 32
                break
        else:
            # Failed to reduce to this rank. 
            # Stop trying lower ranks
            break
            
    # Return the best found
    return {"U": to_list(current_U), "V": to_list(current_V), "W": to_list(current_W)}
