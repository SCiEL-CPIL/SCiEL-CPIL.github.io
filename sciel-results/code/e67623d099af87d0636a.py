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
    Starts from the given U, V, W (which have higher rank).
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W, True

    # We need to eliminate (current_rank - target_rank) columns.
    # Strategy: Zero out columns one by one, optimizing after each zeroing.
    
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    cols_to_remove = current_rank - target_rank
    removed = 0
    
    start_time = time.time()
    
    for i in range(cols_to_remove):
        # Check time
        if time.time() - start_time > time_budget:
            return None, None, None, False
            
        # Choose a column to remove.
        # Heuristic: Remove the column with the smallest norm?
        # Or just pick the last one?
        # Let's pick the column with the smallest combined norm.
        norms = []
        for c in range(U_curr.shape[1]):
            nrm = np.linalg.norm(U_curr[:, c]) + np.linalg.norm(V_curr[:, c]) + np.linalg.norm(W_curr[:, c])
            norms.append(nrm)
        
        # Pick the smallest norm column
        target_col = np.argmin(norms)
        
        # Zero out this column
        U_curr[:, target_col] = 0
        V_curr[:, target_col] = 0
        W_curr[:, target_col] = 0
        
        # Now optimize to restore the tensor with this column effectively zero (or nearly zero)
        # We flatten the matrices, but we must force the target_col entries to be 0.
        # Easier: Just optimize all entries, but initialize with the zeroed column.
        # The optimizer might push it back to non-zero.
        # To prevent this, we can use a penalty or simply remove the column after optimization if it stays small.
        # However, if we remove the column, the rank drops.
        
        # Let's try a different approach:
        # Define a function that takes params for (rank-1) columns.
        # But we are starting from rank R.
        
        # Alternative: Use a penalty term for the target column.
        
        def objective(params):
            U_flat = params[:nm * current_rank]
            V_flat = params[nm * current_rank : nm * current_rank + mp * current_rank]
            W_flat = params[nm * current_rank + mp * current_rank : nm * current_rank + mp * current_rank + pn * current_rank]
            
            U = U_flat.reshape(nm, current_rank)
            V = V_flat.reshape(mp, current_rank)
            W = W_flat.reshape(pn, current_rank)
            
            # Penalty for target column
            pen = 100.0 * (np.sum(U[:, target_col]**2) + np.sum(V[:, target_col]**2) + np.sum(W[:, target_col]**2))
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = M - T_true
            return np.sum(residual**2) + pen
        
        x0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            x_opt = result.x
        except Exception:
            # If optimization fails, keep the zeroed column and move to next?
            # Or break?
            break
            
        U_flat = x_opt[:nm * current_rank]
        V_flat = x_opt[nm * current_rank : nm * current_rank + mp * current_rank]
        W_flat = x_opt[nm * current_rank + mp * current_rank : nm * current_rank + mp * current_rank + pn * current_rank]
        
        U_curr = U_flat.reshape(nm, current_rank)
        V_curr = V_flat.reshape(mp, current_rank)
        W_curr = W_flat.reshape(pn, current_rank)
        
        # Check if the column is effectively zero
        col_norm = np.linalg.norm(U_curr[:, target_col]) + np.linalg.norm(V_curr[:, target_col]) + np.linalg.norm(W_curr[:, target_col])
        
        if col_norm < 1e-3:
            # Remove the column
            U_curr = np.delete(U_curr, target_col, axis=1)
            V_curr = np.delete(V_curr, target_col, axis=1)
            W_curr = np.delete(W_curr, target_col, axis=1)
            current_rank -= 1
            removed += 1
        else:
            # Failed to remove this column. Try another one?
            # For simplicity, break.
            break

    if removed == cols_to_remove:
        # Final polish
        def final_objective(params):
            U_flat = params[:nm * target_rank]
            V_flat = params[nm * target_rank : nm * target_rank + mp * target_rank]
            W_flat = params[nm * target_rank + mp * target_rank : nm * target_rank + mp * target_rank + pn * target_rank]
            
            U = U_flat.reshape(nm, target_rank)
            V = V_flat.reshape(mp, target_rank)
            W = W_flat.reshape(pn, target_rank)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = M - T_true
            return np.sum(residual**2)
            
        x0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
        try:
            result = minimize(final_objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15})
            x_opt = result.x
            U_flat = x_opt[:nm * target_rank]
            V_flat = x_opt[nm * target_rank : nm * target_rank + mp * target_rank]
            W_flat = x_opt[nm * target_rank + mp * target_rank : nm * target_rank + mp * target_rank + pn * target_rank]
            
            U_final = U_flat.reshape(nm, target_rank)
            V_final = V_flat.reshape(mp, target_rank)
            W_final = W_flat.reshape(pn, target_rank)
            
            _, norm = _compute_residual(U_final, V_final, W_final, T_true)
            if norm < 1e-6:
                return U_final, V_final, W_final, True
        except Exception:
            pass
            
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
    
    # Try to reduce to 28
    target_ranks = [31, 30, 29, 28, 27, 26]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        # Calculate time budget for this attempt
        time_left = deadline - time.time()
        budget = min(20.0, time_left * 0.5) # Don't spend too much on one step
        
        if budget < 5.0:
            break
            
        new_U, new_V, new_W, success = _try_reduce_rank(current_U, current_V, current_W, T_true, target_rank, budget)
        
        if success:
            current_U, current_V, current_W = new_U, new_V, new_W
            current_rank = target_rank
            # Check residual
            _, norm = _compute_residual(current_U, current_V, current_W, T_true)
            if norm > 1e-5:
                # Something went wrong, revert
                current_U, current_V, current_W = U, V, W
                current_rank = 32
                break
        else:
            # Failed to reduce to this rank. 
            # We can try a different strategy or stop.
            # For now, stop.
            break
            
    # Return the best found
    return {"U": to_list(current_U), "V": to_list(current_V), "W": to_list(current_W)}
