import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W, rank

def _compute_tensor_fast(U, V, W, n, m, p):
    U_reshaped = U.reshape(n, m, -1) 
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return T

def _verify(U, V, W, n, m, p, tol=1e-6):
    T_decomp = _compute_tensor_fast(U, V, W, n, m, p)
    T_target = np.ones((n, m, p))
    return np.allclose(T_decomp, T_target, atol=tol)

def _try_local_search_drop(U, V, W, n, m, p, max_iter=1000, tol=1e-5, deadline=None):
    """
    Local search to recover the tensor after dropping a column.
    U, V, W are the matrices with the target column zeroed out.
    """
    rank = U.shape[1]
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    T_target = np.ones((n, m, p))
    
    # Initialize with small random noise to escape symmetry
    np.random.seed(42)
    # Add small noise to non-zero entries? Or just perturb all?
    # We only perturb the active columns (all of them, since one is zero)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        T_decomp = _compute_tensor_fast(U, V, W, n, m, p)
        error = np.max(np.abs(T_decomp - T_target))
        
        if error < tol:
            return U, V, W, True
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Gradient-like step or coordinate descent
        # We try to adjust each column slightly
        improved = False
        # Step size decay
        step = 0.1 / (1 + it * 0.01)
        
        for c in range(rank):
            # Skip if column is entirely zero (the dropped one)
            if np.all(U[:, c] == 0) and np.all(V[:, c] == 0) and np.all(W[:, c] == 0):
                continue
                
            # Try perturbing U[:, c]
            for delta in [-step, step]:
                U_old = U[:, c].copy()
                U[:, c] += delta
                T_new = _compute_tensor_fast(U, V, W, n, m, p)
                err_new = np.max(np.abs(T_new - T_target))
                if err_new < error:
                    error = err_new
                    improved = True
                else:
                    U[:, c] = U_old
            
            # Try perturbing V[:, c]
            for delta in [-step, step]:
                V_old = V[:, c].copy()
                V[:, c] += delta
                T_new = _compute_tensor_fast(U, V, W, n, m, p)
                err_new = np.max(np.abs(T_new - T_target))
                if err_new < error:
                    error = err_new
                    improved = True
                else:
                    V[:, c] = V_old
            
            # Try perturbing W[:, c]
            for delta in [-step, step]:
                W_old = W[:, c].copy()
                W[:, c] += delta
                T_new = _compute_tensor_fast(U, V, W, n, m, p)
                err_new = np.max(np.abs(T_new - T_target))
                if err_new < error:
                    error = err_new
                    improved = True
                else:
                    W[:, c] = W_old
        
        if not improved:
            # Try larger random jumps
            for c in range(rank):
                if np.all(U[:, c] == 0) and np.all(V[:, c] == 0) and np.all(W[:, c] == 0):
                    continue
                U_old = U[:, c].copy()
                V_old = V[:, c].copy()
                W_old = W[:, c].copy()
                
                U[:, c] += np.random.normal(0, 0.5, size=U.shape[0])
                V[:, c] += np.random.normal(0, 0.5, size=V.shape[0])
                W[:, c] += np.random.normal(0, 0.5, size=W.shape[0])
                
                T_new = _compute_tensor_fast(U, V, W, n, m, p)
                err_new = np.max(np.abs(T_new - T_target))
                
                if err_new < error:
                    error = err_new
                    improved = True
                else:
                    U[:, c] = U_old
                    V[:, c] = V_old
                    W[:, c] = W_old
            
            if not improved:
                break
    
    T_final = _compute_tensor_fast(U, V, W, n, m, p)
    if np.allclose(T_final, T_target, atol=tol):
        return U, V, W, True
    return best_U, best_V, best_W, False

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), rank_sb
    
    # Try to drop 1 column (Rank 44)
    # We try dropping each column one by one
    for drop_idx in range(rank_sb):
        if time.time() > deadline - 20:
            break
            
        U_test = U_sb.copy()
        V_test = V_sb.copy()
        W_test = W_sb.copy()
        
        # Zero out the column
        U_test[:, drop_idx] = 0.0
        V_test[:, drop_idx] = 0.0
        W_test[:, drop_idx] = 0.0
        
        # Run local search to recover
        U_res, V_res, W_res, success = _try_local_search_drop(
            U_test, V_test, W_test, n, m, p, 
            max_iter=200, 
            deadline=deadline - 20
        )
        
        if success:
            best_U, best_V, best_W, best_rank = U_res, V_res, W_res, rank_sb - 1
            break
    
    # If we didn't find 44, try random search for 44? 
    # Or just return best found.
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
