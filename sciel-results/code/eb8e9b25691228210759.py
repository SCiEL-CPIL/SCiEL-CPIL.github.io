import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

DIMENSIONS = [2, 2, 9]

def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0
    return T

def _compute_residual(U, V, W, T_true):
    """Compute the maximum absolute residual."""
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.max(np.abs(M - T_true))

def _try_rank_direct(rank, T_true, nm, mp, pn, time_budget, n_init=3):
    """Try to find exact rank-rank decomposition via optimization."""
    start_time = time.time()
    best_res = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for init_idx in range(n_init):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization with small values
        scale = 0.1
        U = np.random.randn(nm, rank) * scale
        V = np.random.randn(mp, rank) * scale
        W = np.random.randn(pn, rank) * scale
        
        def objective_and_grad(params):
            U_flat = params[:nm * rank]
            V_flat = params[nm * rank : nm * rank + mp * rank]
            W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            R = M - T_true
            sq_res = np.sum(R**2)
            
            # Gradients
            # grad_U[i,r] = 2 * sum_jk R[i,j,k] V[j,r] W[k,r]
            A = np.einsum('ijk,kr->ijr', R, W)
            grad_U = 2.0 * np.einsum('ijr,jr->ir', A, V)
            
            # grad_V[j,r] = 2 * sum_ik R[i,j,k] U[i,r] W[k,r]
            B = np.einsum('ijk,kr->ijr', R, W)
            grad_V = 2.0 * np.einsum('ijr,ir->jr', B, U)
            
            # grad_W[k,r] = 2 * sum_ij R[i,j,k] U[i,r] V[j,r]
            D = np.einsum('ir,jr->ijr', U, V)
            grad_W = 2.0 * np.einsum('ijr,ijk->kr', D, R)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return sq_res, grad
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxfun': 15000})
            
            U_flat = result.x[:nm * rank]
            V_flat = result.x[nm * rank : nm * rank + mp * rank]
            W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            res = _compute_residual(U, V, W, T_true)
            
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U, V, W
                
            if res < 1e-10:
                return U, V, W, res
                
        except Exception:
            continue
            
    if best_res < 1e-10:
        return best_U, best_V, best_W, best_res
    return None

def _try_rank_reduction(U_base, V_base, W_base, T_true, target_rank, time_budget):
    """Try to reduce rank by finding linear dependencies among columns.
    
    Strategy: from a rank-R decomposition, try to remove 1 or 2 columns
    and re-optimize the remaining columns to still represent the tensor.
    """
    start_time = time.time()
    n_cols = U_base.shape[1]
    nm, mp, pn = U_base.shape[0], V_base.shape[0], W_base.shape[0]
    
    # Try single column removal
    for remove_col in range(n_cols):
        if time.time() - start_time > time_budget:
            break
        if n_cols - 1 < target_rank:
            break
            
        keep = [i for i in range(n_cols) if i != remove_col]
        if len(keep) < target_rank:
            continue
            
        U0 = U_base[:, keep].copy()
        V0 = V_base[:, keep].copy()
        W0 = W_base[:, keep].copy()
        
        # Optimize the reduced decomposition
        def objective_and_grad(params):
            U = params[:nm * len(keep)].reshape(nm, len(keep))
            V = params[nm * len(keep):nm * len(keep) + mp * len(keep)].reshape(mp, len(keep))
            W = params[nm * len(keep) + mp * len(keep):].reshape(pn, len(keep))
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            R = M - T_true
            sq_res = np.sum(R**2)
            
            A = np.einsum('ijk,kr->ijr', R, W)
            grad_U = 2.0 * np.einsum('ijr,jr->ir', A, V)
            B = np.einsum('ijk,kr->ijr', R, W)
            grad_V = 2.0 * np.einsum('ijr,ir->jr', B, U)
            D = np.einsum('ir,jr->ijr', U, V)
            grad_W = 2.0 * np.einsum('ijr,ijk->kr', D, R)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return sq_res, grad
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxfun': 10000})
            
            U = result.x[:nm * len(keep)].reshape(nm, len(keep))
            V = result.x[nm * len(keep):nm * len(keep) + mp * len(keep)].reshape(mp, len(keep))
            W = result.x[nm * len(keep) + mp * len(keep):].reshape(pn, len(keep))
            
            res = _compute_residual(U, V, W, T_true)
            
            if res < 1e-10:
                return U, V, W, res
                
        except Exception:
            continue
    
    # Try pair column removal
    for r1, r2 in combinations(range(n_cols), 2):
        if time.time() - start_time > time_budget:
            break
        if n_cols - 2 < target_rank:
            break
            
        keep = [i for i in range(n_cols) if i != r1 and i != r2]
        if len(keep) < target_rank:
            continue
            
        U0 = U_base[:, keep].copy()
        V0 = V_base[:, keep].copy()
        W0 = W_base[:, keep].copy()
        
        def objective_and_grad(params):
            U = params[:nm * len(keep)].reshape(nm, len(keep))
            V = params[nm * len(keep):nm * len(keep) + mp * len(keep)].reshape(mp, len(keep))
            W = params[nm * len(keep) + mp * len(keep):].reshape(pn, len(keep))
            
            M = np.einsum('ir,jr,kr->ijk', U, V, W)
            R = M - T_true
            sq_res = np.sum(R**2)
            
            A = np.einsum('ijk,kr->ijr', R, W)
            grad_U = 2.0 * np.einsum('ijr,jr->ir', A, V)
            B = np.einsum('ijk,kr->ijr', R, W)
            grad_V = 2.0 * np.einsum('ijr,ir->jr', B, U)
            D = np.einsum('ir,jr->ijr', U, V)
            grad_W = 2.0 * np.einsum('ijr,ijk->kr', D, R)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return sq_res, grad
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxfun': 10000})
            
            U = result.x[:nm * len(keep)].reshape(nm, len(keep))
            V = result.x[nm * len(keep):nm * len(keep) + mp * len(keep)].reshape(mp, len(keep))
            W = result.x[nm * len(keep) + mp * len(keep):].reshape(pn, len(keep))
            
            res = _compute_residual(U, V, W, T_true)
            
            if res < 1e-10:
                return U, V, W, res
                
        except Exception:
            continue
    
    return None

def _build_rank32():
    """Build the known rank-32 decomposition."""
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
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

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T_true = _build_true_tensor(n, m, p)
    
    # Build rank-32 base
    U32, V32, W32 = _build_rank32()
    
    # Try rank reduction from rank-32 base
    remaining = deadline - time.time()
    if remaining > 20:
        budget = min(remaining - 10, 50)
        result = _try_rank_reduction(U32, V32, W32, T_true, 30, budget)
        if result is not None:
            U, V, W, res = result
            if res < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try direct rank 30
    remaining = deadline - time.time()
    if remaining > 25:
        budget = min(remaining - 10, 35)
        result = _try_rank_direct(30, T_true, nm, mp, pn, budget, n_init=4)
        if result is not None:
            U, V, W, res = result
            if res < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank reduction to 31
    remaining = deadline - time.time()
    if remaining > 15:
        budget = min(remaining - 5, 30)
        result = _try_rank_reduction(U32, V32, W32, T_true, 31, budget)
        if result is not None:
            U, V, W, res = result
            if res < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try direct rank 31
    remaining = deadline - time.time()
    if remaining > 15:
        budget = min(remaining - 5, 20)
        result = _try_rank_direct(31, T_true, nm, mp, pn, budget, n_init=3)
        if result is not None:
            U, V, W, res = result
            if res < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 32
    return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
