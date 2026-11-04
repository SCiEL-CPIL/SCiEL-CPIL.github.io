import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid product index."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = np.sum((recon - T) ** 2)
    return err < tol

def _snap_matrix(M, tol=1e-6):
    """Snap matrix entries to nearest integer or half-integer."""
    M_half = np.round(M * 2) / 2.0
    if np.allclose(M, M_half, atol=tol):
        return M_half
    M_int = np.round(M)
    if np.allclose(M, M_int, atol=tol):
        return M_int
    return None

def _reconstruction_error(params, n, m, p, rank, T, penalty_alpha=0.0):
    """Calculate the squared Frobenius norm error + penalty for non-half-integer entries."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U_flat = params[:size_u]
    V_flat = params[size_u:size_u + size_v]
    W_flat = params[size_u + size_v:]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = np.sum((recon - T) ** 2)
    
    if penalty_alpha > 0:
        # Penalty for not being close to half-integers
        # dist(x, k/2) is minimized when x is close to k/2
        # We use squared distance to nearest half-integer
        def half_int_penalty(M):
            nearest = np.round(M * 2) / 2.0
            return np.sum((M - nearest) ** 2)
        
        pen = half_int_penalty(U) + half_int_penalty(V) + half_int_penalty(W)
        err += penalty_alpha * pen
        
    return err

def _search_rank_opt(n, m, p, rank, T, time_budget, seed=42):
    """
    Search for a rank-rank decomposition using gradient-based optimization with annealing.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    max_restarts = 30
    # Allocate time per restart
    time_per_restart = time_budget / max_restarts
    
    best_error = np.inf
    best_params = None
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Random initialization with small variance
        scale = 0.5
        U0 = np.random.randn(n * m, rank) * scale
        V0 = np.random.randn(m * p, rank) * scale
        W0 = np.random.randn(p * n, rank) * scale
        
        # Normalize to avoid scale drift issues during optimization
        # Fix scale of U to be 1 (if non-zero)
        if np.linalg.norm(U0) > 1e-6:
            U0 = U0 / np.linalg.norm(U0)
            V0 = V0 * np.linalg.norm(U0) # This logic is tricky, better to just let it optimize and snap later
            # Actually, let's just optimize freely and rely on the penalty/snapping.
            # The scale invariance means the error surface has a valley. 
            # We can add a constraint ||U||_F = 1 to break symmetry.
            # Let's try without constraint first, but with a small regularization on norms.
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Define objective with annealing penalty
        # Start with low penalty, increase it
        alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
        
        current_params = x0.copy()
        
        for alpha in alphas:
            if time.time() - start_time > time_budget:
                break
                
            def objective(x):
                return _reconstruction_error(x, n, m, p, rank, T, penalty_alpha=alpha)
            
            try:
                res = minimize(objective, current_params, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6})
                current_params = res.x
            except Exception:
                continue
            
            # Check if we are close to a solution
            err_no_pen = _reconstruction_error(current_params, n, m, p, rank, T, penalty_alpha=0.0)
            if err_no_pen < best_error:
                best_error = err_no_pen
                best_params = current_params.copy()
                
            # Early termination if exact solution found (within tolerance for snapping)
            if err_no_pen < 1e-4:
                # Try to snap
                size_u = n * m * rank
                size_v = m * p * rank
                U_snap = _snap_matrix(current_params[:size_u].reshape(n * m, rank))
                V_snap = _snap_matrix(current_params[size_u:size_u+size_v].reshape(m * p, rank))
                W_snap = _snap_matrix(current_params[size_u+size_v:].reshape(p * n, rank))
                
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if _check_decomposition(U_snap, V_snap, W_snap, T):
                        return U_snap, V_snap, W_snap

    # Final check on best_params
    if best_params is not None:
        size_u = n * m * rank
        size_v = m * p * rank
        U_snap = _snap_matrix(best_params[:size_u].reshape(n * m, rank))
        V_snap = _snap_matrix(best_params[size_u:size_u+size_v].reshape(m * p, rank))
        W_snap = _snap_matrix(best_params[size_u+size_v:].reshape(p * n, rank))
        
        if U_snap is not None and V_snap is not None and W_snap is not None:
            if _check_decomposition(U_snap, V_snap, W_snap, T):
                return U_snap, V_snap, W_snap

    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    total_start = time.time()
    time_budget = 140.0
    
    # Fallback: Schoolbook Rank 24
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    # Try ranks from 19 up to 23
    # Rank 19 is the known optimal for 2x3x4
    for rank in [19, 20, 21, 22, 23]:
        if rank > rank_fb:
            continue
            
        elapsed = time.time() - total_start
        remaining = time_budget - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time
        if rank == 19:
            rank_budget = remaining * 0.8
        elif rank == 20:
            rank_budget = remaining * 0.15
        else:
            rank_budget = remaining * 0.05
            
        result = _search_rank_opt(n, m, p, rank, T, rank_budget)
        
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
# EVOLVE-BLOCK-END
