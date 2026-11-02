import numpy as np
import time
import itertools
from scipy.optimize import minimize

def construct():
    """
    Attempt to find a rank-minimal decomposition for 3x3 * 3x5.
    Uses structured search with fallback to schoolbook (rank 45).
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin

    n, m, p = 3, 3, 5
    # Indices:
    # U: (n*m, R) -> 9 x R
    # V: (m*p, R) -> 15 x R
    # W: (p*n, R) -> 15 x R
    # Note: W index is k*n + i.
    
    # Target tensor T of shape (9, 15, 15)
    T = np.zeros((9, 15, 15), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def compute_tensor(U, V, W):
        """Compute the tensor from factor matrices."""
        R = U.shape[1]
        T_comp = np.zeros((9, 15, 15), dtype=np.float64)
        for r in range(R):
            T_comp += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        return T_comp

    def residual_norm(U, V, W):
        """Compute L2 norm of residual."""
        T_comp = compute_tensor(U, V, W)
        return np.linalg.norm(T_comp - T)

    def objective(x, R):
        """Objective function for optimization."""
        U = x[:9*R].reshape(9, R)
        V = x[9*R:9*R+15*R].reshape(15, R)
        W = x[9*R+15*R:].reshape(15, R)
        T_comp = compute_tensor(U, V, W)
        return np.sum((T_comp - T)**2)

    def gradient(x, R):
        """Gradient of the objective function."""
        U = x[:9*R].reshape(9, R)
        V = x[9*R:9*R+15*R].reshape(15, R)
        W = x[9*R+15*R:].reshape(15, R)
        
        T_comp = compute_tensor(U, V, W)
        diff = T_comp - T
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(R):
            grad_U[:, r] = np.einsum('uvw,vw->u', diff, V[:, r] * W[:, r])
            grad_V[:, r] = np.einsum('uvw,uw->v', diff, U[:, r] * W[:, r])
            grad_W[:, r] = np.einsum('uvw,uv->w', diff, U[:, r] * V[:, r])
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def try_rank(R, max_restarts=3, max_iters=50):
        """Try to find a rank-R decomposition."""
        if time.time() > deadline:
            return None
        
        best_x = None
        best_obj = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            x0 = np.random.randn(9*R + 15*R + 15*R) * 0.3
            
            try:
                result = minimize(
                    objective,
                    x0,
                    args=(R,),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': max_iters, 'gtol': 1e-6}
                )
                
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_x = result.x
                    
            except Exception as e:
                continue
        
        if best_x is None or best_obj > 1e-4:
            return None
        
        # Extract factors
        U = best_x[:9*R].reshape(9, R)
        V = best_x[9*R:9*R+15*R].reshape(15, R)
        W = best_x[9*R+15*R:].reshape(15, R)
        
        # Snap to nearest half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check if snapped version still works
        if residual_norm(U_snap, V_snap, W_snap) < 1e-4:
            return (U_snap, V_snap, W_snap)
        
        # Try integer snapping
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        if residual_norm(U_int, V_int, W_int) < 1e-4:
            return (U_int, V_int, W_int)
        
        # Return the continuous solution if it's close enough
        if best_obj < 1e-2:
            return (U, V, W)
        
        return None

    # Try decreasing ranks
    for R in range(10, 25):
        if time.time() > deadline:
            break
            
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
