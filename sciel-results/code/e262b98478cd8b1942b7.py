# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 4, 8]

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] using einsum."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    return np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

def _verify_decomposition(decomp, T, tol=1e-9):
    """Check if the decomposition exactly reproduces T."""
    U, V, W = decomp['U'], decomp['V'], decomp['W']
    T_approx = _tensor_from_decomp(U, V, W)
    return np.allclose(T_approx, T, atol=tol, rtol=0)

def _loss_and_grad(U, V, W, T):
    """
    Compute loss ||T - sum_r U_r V_r W_r||^2 and gradients dU, dV, dW.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    T: (n*m, m*p, p*n)
    """
    T_pred = np.einsum('ar,br,cr->abc', U, V, W)
    diff = T - T_pred
    
    loss = np.sum(diff ** 2)
    
    dU = -2 * np.einsum('abc,br,cr->ar', diff, V, W)
    dV = -2 * np.einsum('abc,ar,cr->br', diff, U, W)
    dW = -2 * np.einsum('abc,ar,br->cr', diff, U, V)
    
    return loss, dU, dV, dW

def _optimize_decomposition_l_bfgs(n, m, p, T, R, max_iters=200, deadline=None):
    """
    Optimize decomposition of rank R using L-BFGS-B.
    """
    # Flatten U, V, W into a single vector
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Initialize with random values
    np.random.seed(42)
    U0 = np.random.randn(dim_U, R) * 0.5
    V0 = np.random.randn(dim_V, R) * 0.5
    W0 = np.random.randn(dim_W, R) * 0.5
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return 1e10, np.zeros_like(x)
            
        U = x[:dim_U * R].reshape(dim_U, R)
        V = x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W = x[dim_U * R + dim_V * R :].reshape(dim_W, R)
        
        loss, dU, dV, dW = _loss_and_grad(U, V, W, T)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad

    # Use L-BFGS-B
    res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', jac=True, 
                                   options={'maxiter': max_iters, 'disp': False})
    
    if res.success or res.fun < 1e-8:
        U_opt = res.x[:dim_U * R].reshape(dim_U, R)
        V_opt = res.x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W_opt = res.x[dim_U * R + dim_V * R :].reshape(dim_W, R)
        return U_opt, V_opt, W_opt, res.fun
    else:
        return None, None, None, res.fun

def _snap_to_half_integers(U, V, W, tol=1e-6):
    """
    Snap values to nearest half-integer if close enough.
    """
    def snap(arr):
        arr_2 = arr * 2
        arr_rounded = np.round(arr_2)
        if np.allclose(arr_2, arr_rounded, atol=tol):
            return arr_rounded / 2.0
        else:
            return arr

    U_snapped = snap(U)
    V_snapped = snap(V)
    W_snapped = snap(W)
    
    T_pred = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
    
    return U_snapped, V_snapped, W_snapped, T_pred

def _try_find_smaller_decomposition(n, m, p, deadline):
    """
    Search for a decomposition with rank < n*m*p.
    """
    schoolbook_rank = n * m * p
    T = _build_target_tensor(n, m, p)
    
    best_rank = schoolbook_rank
    best_decomp = _schoolbook(n, m, p)
    
    # Try ranks from schoolbook_rank - 1 down to a limit
    # We will try a few ranks below schoolbook
    ranks_to_try = []
    for R in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 1), -1):
        ranks_to_try.append(R)
        
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Slightly different initialization for each restart
            np.random.seed(42 + restart)
            # Use L-BFGS-B
            U_opt, V_opt, W_opt, loss = _optimize_decomposition_l_bfgs(n, m, p, T, R, max_iters=100, deadline=deadline)
            
            if U_opt is not None and loss < 1e-8:
                # Try to snap to half-integers
                U_s, V_s, W_s, T_pred = _snap_to_half_integers(U_opt, V_opt, W_opt)
                if np.allclose(T_pred, T, atol=1e-9, rtol=0):
                    decomp = {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
                    return R, decomp
                    found = True
                    break
                    
    return best_rank, best_decomp

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s buffer for the 150s limit
    
    # Try to find a smaller decomposition
    rank, decomp = _try_find_smaller_decomposition(n, m, p, deadline)
    
    # Verify the decomposition to ensure it is correct
    T = _build_target_tensor(n, m, p)
    if not _verify_decomposition(decomp, T):
        # Fallback to schoolbook if verification fails
        decomp = _schoolbook(n, m, p)
    
    return decomp
# EVOLVE-BLOCK-END
