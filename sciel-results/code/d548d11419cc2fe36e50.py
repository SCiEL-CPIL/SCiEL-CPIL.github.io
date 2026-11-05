import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    # First, try to find a rank-16 decomposition
    result_16 = _try_rank16(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Fallback: rank-17 decomposition
    return _rank17_decomposition(n, m, p)

def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # Block 1: Strassen 2x2x2 (j in {0,1}, k in {0,1})
    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    # Block 2: Direct 2x2x1 (j in {0,1}, k=2)
    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    # Block 3: Direct 2x1x3 (j=2, k in {0,1,2})
    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }

def _check_exact(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition is exact (half-integer entries)."""
    # Round to half-integers
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    # Check if the rounded values actually produce the tensor
    # Convert to integer grid (multiply by 2) for exact integer arithmetic check
    Ui = np.rint(U_s * 2).astype(int)
    Vi = np.rint(V_s * 2).astype(int)
    Wi = np.rint(W_s * 2).astype(int)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    # The product in the integer domain will be 4 * (original product)
    # Original product should be 1. So integer product should be 4.
    # Wait, U, V, W are half-integers.
    # Let U = Ui/2, V = Vi/2, W = Wi/2.
    # Sum (Ui/2) (Vi/2) (Wi/2) = 1/8 Sum Ui Vi Wi.
    # So Sum Ui Vi Wi should be 8.
    
    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None

def _try_rank16(n, m, p, deadline):
    rank = 16
    
    # Get the rank 17 solution to use as a base for perturbation
    base_17 = _rank17_decomposition(n, m, p)
    U_base = np.array(base_17["U"])
    V_base = np.array(base_17["V"])
    W_base = np.array(base_17["W"])
    
    best_err = float('inf')
    best_state = None
    
    # Strategy 1: Try structured initializations from rank-17 by merging columns
    # Strategy 2: Random initializations with better optimization
    
    for seed in range(500):
        if time.time() > deadline:
            break
            
        np.random.seed(seed)
        
        # Different initialization strategies
        if seed < 100:
            # Strategy 1: Perturb rank-17, drop one column, optimize
            U = U_base[:, :rank].copy()
            V = V_base[:, :rank].copy()
            W = W_base[:, :rank].copy()
            # Add small perturbation
            U += np.random.randn(n * m, rank) * 0.1
            V += np.random.randn(m * p, rank) * 0.1
            W += np.random.randn(p * n, rank) * 0.1
        elif seed < 200:
            # Strategy 2: Scale and perturb
            U = U_base[:, :rank].copy() * (0.9 + 0.2 * np.random.randn())
            V = V_base[:, :rank].copy() * (0.9 + 0.2 * np.random.randn())
            W = W_base[:, :rank].copy() * (0.9 + 0.2 * np.random.randn())
        elif seed < 300:
            # Strategy 3: Random with structure
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            # Add some structure: make some entries zero
            for r in range(rank):
                if np.random.rand() < 0.3:
                    U[np.random.randint(n*m), r] = 0
                if np.random.rand() < 0.3:
                    V[np.random.randint(m*p), r] = 0
                if np.random.rand() < 0.3:
                    W[np.random.randint(p*n), r] = 0
        else:
            # Strategy 4: Larger random
            U = np.random.randn(n * m, rank) * 1.0
            V = np.random.randn(m * p, rank) * 1.0
            W = np.random.randn(p * n, rank) * 1.0
        
        # Use scipy.optimize for better convergence
        def objective(x):
            U_t = x[:n*m*rank].reshape(n, m, rank)
            V_t = x[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
            W_t = x[n*m*rank+m*p*rank:].reshape(p, n, rank)
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            return np.sum(res**2)
        
        def gradient(x):
            U_t = x[:n*m*rank].reshape(n, m, rank)
            V_t = x[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
            W_t = x[n*m*rank+m*p*rank:].reshape(p, n, rank)
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t).ravel()
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t).ravel()
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t).ravel()
            return np.concatenate([dU, dV, dW])
        
        x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        
        try:
            res_opt = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res_opt.fun < best_err:
                best_err = res_opt.fun
                x_best = res_opt.x
                U_b = x_best[:n*m*rank].reshape(n*m, rank)
                V_b = x_best[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_b = x_best[n*m*rank+m*p*rank:].reshape(p*n, rank)
                best_state = (U_b, V_b, W_b)
                
                if best_err < 1e-8:
                    ok, Us, Vs, Ws = _check_exact(U_b, V_b, W_b, n, m, p)
                    if ok:
                        return {
                            "U": Us.tolist(),
                            "V": Vs.tolist(),
                            "W": Ws.tolist()
                        }
        except Exception:
            pass
        
        # Check best state periodically
        if best_state is not None and seed % 10 == 0:
            U_b, V_b, W_b = best_state
            ok, Us, Vs, Ws = _check_exact(U_b, V_b, W_b, n, m, p)
            if ok:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
        
        # Try Newton polish on best state
        if best_state is not None and best_err < 1e-3 and seed % 5 == 0:
            U_b, V_b, W_b = best_state
            U_p, V_p, W_p, success = _newton_polish(U_b, V_b, W_b, n, m, p, max_iter=100, tol=1e-14)
            if success:
                ok, Us, Vs, Ws = _check_exact(U_p, V_p, W_p, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
            
            # Also check if the best state itself is close to exact
            ok, Us, Vs, Ws = _check_exact(U_b, V_b, W_b, n, m, p)
            if ok:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
                
    return None

def _newton_polish(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """
    Newton-type polish for the tensor decomposition.
    """
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    for it in range(max_iter):
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - 1.0
        err = np.sum(res**2)
        
        if err < tol:
            return U, V, W, True
        
        # Gauss-Newton step
        dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
        dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
        dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
        
        # Adaptive step size
        alpha = 1.0
        for _ in range(20):
            U_new = U - alpha * dU.reshape(n*m, rank)
            V_new = V - alpha * dV.reshape(m*p, rank)
            W_new = W - alpha * dW.reshape(p*n, rank)
            
            U_t_new = U_new.reshape(n, m, rank)
            V_t_new = V_new.reshape(m, p, rank)
            W_t_new = W_new.reshape(p, n, rank)
            
            prod_new = np.einsum('ijr,jkr,kir->ijk', U_t_new, V_t_new, W_t_new)
            res_new = prod_new - 1.0
            err_new = np.sum(res_new**2)
            
            if err_new < err:
                break
            alpha *= 0.5
        
        U = U_new
        V = V_new
        W = W_new
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
    
    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = np.sum((prod - 1.0)**2)
    return U, V, W, err < 1e-8
