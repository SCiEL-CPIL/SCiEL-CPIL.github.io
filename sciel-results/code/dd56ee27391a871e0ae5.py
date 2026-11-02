"""Program for matrix multiplication 2x3 by 3x4 using numerical optimization."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Uses numerical optimization (L-BFGS-B style via scipy) to search for low-rank
    decompositions. Tries ranks from theoretical lower bound upward.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    start_time = time.time()
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But we need to check if rank 12 is achievable.
    for rank in range(12, 25):
        if time.time() - start_time > _DEADLINE - 10:
            break
        result = _solve_rank(rank, n, m, p, start_time)
        if result is not None:
            return result
    
    # Fallback to schoolbook
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


def _solve_rank(rank, n, m, p, start_time):
    """
    Try to find a rank-r decomposition using numerical optimization with vectorized operations.
    """
    from scipy.optimize import minimize
    
    num_um = n * m
    num_vm = m * p
    num_wm = p * n
    total_vars = rank * (num_um + num_vm + num_wm)
    
    # Construct target tensor
    T_target = np.zeros((num_um, num_vm, num_wm))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def _unpack(params):
        U = params[:rank * num_um].reshape(rank, num_um).T
        V = params[rank * num_um:rank * (num_um + num_vm)].reshape(rank, num_vm).T
        W = params[rank * (num_um + num_vm):].reshape(rank, num_wm).T
        return U, V, W

    def _objective(params):
        U, V, W = _unpack(params)
        T_comp = np.einsum('uc,vc,wc->uvw', U, V, W)
        diff = T_comp - T_target
        return np.sum(diff * diff)

    def _gradient(params):
        U, V, W = _unpack(params)
        T_comp = np.einsum('uc,vc,wc->uvw', U, V, W)
        diff = T_comp - T_target
        
        grad_U = np.einsum('uvw,vc,wc->uc', diff, V, W)
        grad_V = np.einsum('uvw,uc,wc->vc', diff, U, W)
        grad_W = np.einsum('uvw,uc,vc->wc', diff, U, V)
        
        grad = np.concatenate([
            grad_U.T.flatten(),
            grad_V.T.flatten(),
            grad_W.T.flatten()
        ])
        return 2 * grad

    np.random.seed(42)
    best_obj = np.inf
    
    max_starts = 100
    for start_idx in range(max_starts):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        params0 = np.random.randn(total_vars) * 1.0
        
        try:
            res = minimize(
                _objective,
                params0,
                jac=_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                if res.fun < 1e-10:
                    U, V, W = _unpack(res.x)
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if _verify(U_r, V_r, W_r, n, m, p):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
        except Exception:
            continue
            
    return None


def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    num_um = n * m
    num_vm = m * p
    num_wm = p * n
    
    T_comp = np.einsum('uc,vc,wc->uvw', U, V, W)
    
    T_target = np.zeros((num_um, num_vm, num_wm))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
                
    if np.max(np.abs(T_comp - T_target)) > tol:
        return False
        
    return True


# EVOLVE-BLOCK-END
