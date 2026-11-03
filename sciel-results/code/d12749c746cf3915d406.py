"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 120.0
_SAFETY_MARGIN = 15.0

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition."""
    rank = U.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                error += (val - 1.0) ** 2
    return error

def _als_step(U, V, W, n, m, p, rank):
    """Perform one ALS iteration to minimize the error."""
    # Update U
    for i in range(n):
        for j in range(m):
            # Solve for U[i*m+j, :]
            # The contribution to the error from U[i*m+j, c] is:
            # sum_{k} (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1)^2
            # We can rewrite this as a linear system in U[i*m+j, :]
            # Let x = U[i*m+j, :]
            # Error term: sum_k (sum_c x_c V[j*p+k,c] W[k*n+i,c] - 1)^2
            # This is a quadratic in x. The gradient is:
            # 2 * sum_k (sum_c x_c V[j*p+k,c] W[k*n+i,c] - 1) * V[j*p+k,c] * W[k*n+i,c]
            # Setting gradient to 0:
            # sum_k (sum_c x_c V[j*p+k,c] W[k*n+i,c]) V[j*p+k,c'] W[k*n+i,c'] = sum_k V[j*p+k,c'] W[k*n+i,c']
            # for each c'
            
            # Let's form the normal equations:
            # A x = b where A[c, c'] = sum_k V[j*p+k,c] W[k*n+i,c] V[j*p+k,c'] W[k*n+i,c']
            # b[c'] = sum_k V[j*p+k,c'] W[k*n+i,c']
            
            A = np.zeros((rank, rank))
            b = np.zeros(rank)
            for k in range(p):
                v_vec = V[j*p+k, :]
                w_vec = W[k*n+i, :]
                A += np.outer(v_vec * w_vec, v_vec * w_vec)
                b += v_vec * w_vec
            
            # Add regularization for stability
            A += 1e-10 * np.eye(rank)
            
            try:
                U[i*m+j, :] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                pass
    
    # Update V
    for j in range(m):
        for k in range(p):
            A = np.zeros((rank, rank))
            b = np.zeros(rank)
            for i in range(n):
                u_vec = U[i*m+j, :]
                w_vec = W[k*n+i, :]
                A += np.outer(u_vec * w_vec, u_vec * w_vec)
                b += u_vec * w_vec
            
            A += 1e-10 * np.eye(rank)
            
            try:
                V[j*p+k, :] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                pass
    
    # Update W
    for k in range(p):
        for i in range(n):
            A = np.zeros((rank, rank))
            b = np.zeros(rank)
            for j in range(m):
                u_vec = U[i*m+j, :]
                v_vec = V[j*p+k, :]
                A += np.outer(u_vec * v_vec, u_vec * v_vec)
                b += u_vec * v_vec
            
            A += 1e-10 * np.eye(rank)
            
            try:
                W[k*n+i, :] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                pass

def _try_find_decomposition(rank_target, n, m, p, max_iterations=1000, num_restarts=5):
    """Try to find a decomposition of the given rank using ALS."""
    if rank_target < 1:
        return None
    
    best_solution = None
    best_error = float('inf')
    
    for restart in range(num_restarts):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
        
        # Random initialization
        U = np.random.randn(n*m, rank_target) * 0.5
        V = np.random.randn(m*p, rank_target) * 0.5
        W = np.random.randn(p*n, rank_target) * 0.5
        
        current_error = _compute_error(U, V, W, n, m, p)
        
        for it in range(max_iterations):
            if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
                break
            
            # Perform one ALS iteration
            _als_step(U, V, W, n, m, p, rank_target)
            
            new_error = _compute_error(U, V, W, n, m, p)
            
            if new_error < 1e-12:
                # Found exact solution
                return (U, V, W)
            
            if new_error < best_error:
                best_error = new_error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            # Check for convergence
            if abs(current_error - new_error) < 1e-10:
                break
            
            current_error = new_error
    
    if best_solution is not None and best_error < 1e-8:
        return best_solution
    
    return None

def _try_l1_pruning(U, V, W, n, m, p, target_rank, max_iterations=500):
    """Try to reduce rank using L1 regularization and pruning."""
    rank = U.shape[1]
    if rank <= target_rank:
        return (U, V, W)
    
    # Compute L1 norms of columns
    U_l1 = np.sum(np.abs(U), axis=0)
    V_l1 = np.sum(np.abs(V), axis=0)
    W_l1 = np.sum(np.abs(W), axis=0)
    
    # Combine norms
    combined_l1 = U_l1 + V_l1 + W_l1
    
    # Sort by L1 norm (smallest first)
    sorted_indices = np.argsort(combined_l1)
    
    # Try to prune columns with smallest L1 norms
    for prune_count in range(rank - target_rank + 1):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
        
        # Keep the largest (rank - prune_count) columns
        keep_count = rank - prune_count
        if keep_count < target_rank:
            break
        
        keep_indices = sorted_indices[rank - keep_count:]
        
        U_pruned = U[:, keep_indices]
        V_pruned = V[:, keep_indices]
        W_pruned = W[:, keep_indices]
        
        error = _compute_error(U_pruned, V_pruned, W_pruned, n, m, p)
        
        if error < 1e-8:
            # Try to refine
            refined = _try_find_decomposition(keep_count, n, m, p, max_iterations=200, num_restarts=1)
            if refined is not None:
                return refined
            else:
                return (U_pruned, V_pruned, W_pruned)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # Start with a reasonable rank and try to reduce
    # Lower bound is max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    lower_bound = max(n * m, m * p, p * n)
    
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try ranks from lower_bound upward
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
            break
        
        # Try to find a decomposition of this rank
        result = _try_find_decomposition(rank, n, m, p, max_iterations=500, num_restarts=3)
        
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            # Round to nearest 0.5 for cleaner output
            U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
            V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
            W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
            
            if _validate(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try L1 pruning from schoolbook
    if time.time() - _START_TIME < _TIMEOUT - _SAFETY_MARGIN:
        U_sb = np.array(best_result["U"])
        V_sb = np.array(best_result["V"])
        W_sb = np.array(best_result["W"])
        
        # Try to prune to various ranks
        for target_rank in range(lower_bound, schoolbook_rank):
            if time.time() - _START_TIME > _TIMEOUT - _SAFETY_MARGIN:
                break
            
            pruned = _try_l1_pruning(U_sb, V_sb, W_sb, n, m, p, target_rank)
            if pruned is not None:
                U, V, W = pruned
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                if _validate(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    return best_result
# EVOLVE-BLOCK-END
