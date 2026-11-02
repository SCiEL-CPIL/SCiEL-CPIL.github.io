import time
import numpy as np
from scipy.optimize import minimize

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

def _try_reduce_rank(U, V, W, T_true, target_rank, time_budget, seed=0):
    """
    Try to reduce the rank of the decomposition to target_rank.
    Uses a penalty method to force specific columns to zero.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W, True

    cols_to_remove = current_rank - target_rank
    start_time = time.time()
    
    # We will try to remove columns one by one.
    # To increase success rate, we try different orders of removal.
    # We use a simple heuristic: remove the column with the smallest norm first,
    # but we also try random orders.
    
    best_result = None
    best_norm = float('inf')
    
    # Number of random restarts/orders to try
    num_restarts = 3
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
            
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        current_rank_curr = current_rank
        removed = 0
        
        # Generate a random permutation of column indices to try removing
        # We will iterate through this list and try to remove each one
        # that is still present.
        if restart == 0:
            # Heuristic: Smallest norm
            norms = [np.linalg.norm(U_curr[:, c]) + np.linalg.norm(V_curr[:, c]) + np.linalg.norm(W_curr[:, c]) for c in range(current_rank_curr)]
            order = np.argsort(norms)
        else:
            order = np.random.permutation(current_rank_curr)
        
        for target_col in order:
            if time.time() - start_time > time_budget:
                break
                
            if removed >= cols_to_remove:
                break
                
            # Check if this column is still in the current set
            # We are deleting columns, so indices shift. 
            # To avoid complexity, we will zero out a column in the full matrix, 
            # then delete it.
            # But 'target_col' is an index in the ORIGINAL or CURRENT matrix?
            # In the current loop, we are iterating over the original indices.
            # If we delete a column, the indices of subsequent columns change.
            # This is tricky. 
            # Better approach: Work with a mask or just zero out and delete.
            
            # Let's stick to the previous approach: 
            # We have a matrix of current_rank_curr columns.
            # We pick a column index in the CURRENT matrix.
            # If we pick from the original order, we need to map it.
            
            # Simpler: Just pick a random column from the CURRENT matrix each time.
            # But we want to try specific ones.
            
            # Let's restart the logic for clarity:
            # We will try to remove `cols_to_remove` columns.
            # In each step, we pick a column from the current set to zero out.
            
            # To make it deterministic and varied by restart:
            # We will use the 'order' list to pick which column to target.
            # If the column was already removed, we skip to the next in order.
            
            # This is getting complicated. Let's use a simpler strategy:
            # Just try to remove one column at a time, picking the one with the smallest norm in the CURRENT matrix.
            # This is what the original code did, but it failed.
            # Why? Maybe the optimization didn't converge.
            
            # Let's try a different penalty: 
            # Minimize ||M - T||^2 + lambda * ||Col||^2
            # With a large lambda.
            
            # Let's try to remove the column with the smallest norm in the current matrix.
            norms_curr = [np.linalg.norm(U_curr[:, c]) + np.linalg.norm(V_curr[:, c]) + np.linalg.norm(W_curr[:, c]) for c in range(current_rank_curr)]
            target_col_idx = np.argmin(norms_curr)
            
            # Objective function
            def objective(params, target_col_idx=target_col_idx, curr_rank=current_rank_curr):
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
                col_penalty = 100.0 * (np.sum(U_mat[:, target_col_idx]**2) + 
                                      np.sum(V_mat[:, target_col_idx]**2) + 
                                      np.sum(W_mat[:, target_col_idx]**2))
                
                return base_loss + col_penalty
            
            x0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
                x_opt = result.x
            except Exception:
                continue
                
            U_flat = x_opt[:nm * current_rank_curr]
            V_flat = x_opt[nm * current_rank_curr : nm * current_rank_curr + mp * current_rank_curr]
            W_flat = x_opt[nm * current_rank_curr + mp * current_rank_curr : nm * current_rank_curr + mp * current_rank_curr + pn * current_rank_curr]
            
            U_curr = U_flat.reshape(nm, current_rank_curr)
            V_curr = V_flat.reshape(mp, current_rank_curr)
            W_curr = W_flat.reshape(pn, current_rank_curr)
            
            # Check if column is effectively zero
            col_norm = np.linalg.norm(U_curr[:, target_col_idx]) + np.linalg.norm(V_curr[:, target_col_idx]) + np.linalg.norm(W_curr[:, target_col_idx])
            
            if col_norm < 1e-4:
                U_curr = np.delete(U_curr, target_col_idx, axis=1)
                V_curr = np.delete(V_curr, target_col_idx, axis=1)
                W_curr = np.delete(W_curr, target_col_idx, axis=1)
                current_rank_curr -= 1
                removed += 1
            else:
                # If we can't remove this column, try another one?
                # For now, break and try a different restart order.
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
                result = minimize(final_objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
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
        # Round to nearest half-integer if close
        # The task requires integer/half-integer entries
        # We will round to the nearest 0.5
        arr = np.round(arr * 2) / 2
        return arr.tolist()
    
    current_U, current_V, current_W = U, V, W
    current_rank = 32
    
    # Try to reduce step by step
    # We prioritize finding rank 29 or 30.
    target_ranks = [31, 30, 29]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Calculate time budget for this attempt
        time_left = deadline - time.time()
        budget = min(30.0, time_left * 0.5) # Don't spend too much on one step
        
        if budget < 10.0:
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
