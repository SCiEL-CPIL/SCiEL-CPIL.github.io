import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _mismatch_count(U, V, W, n, m, p, tol=1e-6):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.einsum('it,jt,kt->ijk', Ua, Va, Wa)
    target = np.ones((n, m, p))
    return int(np.sum(np.abs(T - target) > tol))

def _verify(U, V, W, n, m, p, atol=1e-9):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.einsum('it,jt,kt->ijk', Ua, Va, Wa)
    target = np.ones((n, m, p))
    return bool(np.allclose(T, target, atol=atol))

def _local_polish(U, V, W, n, m, p, max_iter=2000):
    """Try small half-integer moves to reduce mismatch count."""
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    cur = _mismatch_count(Ua, Va, Wa, n, m, p)
    if cur == 0:
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    r = Ua.shape[1]
    all_vars = [Ua, Va, Wa]
    for _ in range(max_iter):
        improved = False
        for mat in all_vars:
            for idx in range(mat.size):
                flat = mat.reshape(-1)
                base = flat[idx]
                for delta in (-0.5, 0.5):
                    new_val = base + delta
                    # snap to half-integer
                    new_val = np.round(new_val * 2) / 2
                    if new_val == base:
                        continue
                    flat[idx] = new_val
                    cnt = _mismatch_count(Ua, Va, Wa, n, m, p)
                    if cnt < cur:
                        cur = cnt
                        improved = True
                        if cur == 0:
                            return Ua.tolist(), Va.tolist(), Wa.tolist()
                        break
                    else:
                        flat[idx] = base
        if not improved:
            break
    if cur == 0:
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    return None

def _structured_init(n, m, p, r, rng):
    """Generate structured initial guesses exploiting matrix multiplication structure."""
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Initialize with random half-integer values
    U[:] = rng.randint(-3, 4, size=(n * m, r)) * 0.5
    V[:] = rng.randint(-3, 4, size=(m * p, r)) * 0.5
    W[:] = rng.randint(-3, 4, size=(p * n, r)) * 0.5
    
    # Add some structure: ensure each row has at least one non-zero
    for i in range(n * m):
        if np.all(U[i] == 0):
            U[i, rng.randint(r)] = 0.5 * (1 if rng.rand() < 0.5 else -1)
    for j in range(m * p):
        if np.all(V[j] == 0):
            V[j, rng.randint(r)] = 0.5 * (1 if rng.rand() < 0.5 else -1)
    for k in range(p * n):
        if np.all(W[k] == 0):
            W[k, rng.randint(r)] = 0.5 * (1 if rng.rand() < 0.5 else -1)
    
    return U, V, W

def _objective_with_grad(params, r, n, m, p):
    """Compute objective and gradient for L-BFGS-B."""
    U = params[:n * m * r].reshape(n * m, r)
    V = params[n * m * r : n * m * r + m * p * r].reshape(m * p, r)
    W = params[n * m * r + m * p * r :].reshape(p * n, r)
    
    T = np.einsum('it,jt,kt->ijk', U, V, W)
    diff = T - 1.0
    
    # Objective
    obj = float(np.sum(diff * diff))
    
    # Gradient computation
    # d(obj)/d(U[i,t]) = 2 * sum_{j,k} diff[i,j,k] * V[j,t] * W[k,t]
    # d(obj)/d(V[j,t]) = 2 * sum_{i,k} diff[i,j,k] * U[i,t] * W[k,t]
    # d(obj)/d(W[k,t]) = 2 * sum_{i,j} diff[i,j,k] * U[i,t] * V[j,t]
    
    U_grad = 2.0 * np.einsum('ijk,jt,kt->it', diff, V, W)
    V_grad = 2.0 * np.einsum('ijk,it,kt->jt', diff, U, W)
    W_grad = 2.0 * np.einsum('ijk,it,jt->kt', diff, U, V)
    
    grad = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
    
    return obj, grad

def _try_rank_optimized(r, n_restarts, time_limit, start_time, n, m, p):
    """Try to find rank-r decomposition using optimized search."""
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            return None
        
        rng = np.random.RandomState(restart * 104729 + r * 31337)
        
        # Use structured initialization
        U0, V0, W0 = _structured_init(n, m, p, r, rng)
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            from scipy.optimize import minimize
            
            # Use L-BFGS-B with gradient
            res = minimize(
                lambda x: _objective_with_grad(x, r, n, m, p)[0],
                x0,
                jac=lambda x: _objective_with_grad(x, r, n, m, p)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() - start_time > time_limit:
                return None
            
            if res.fun < 1e-3:
                Uo = res.x[:n * m * r].reshape(n * m, r)
                Vo = res.x[n * m * r : n * m * r + m * p * r].reshape(m * p, r)
                Wo = res.x[n * m * r + m * p * r :].reshape(p * n, r)
                
                # Round to half-integers
                Ui = np.round(Uo * 2) / 2
                Vi = np.round(Vo * 2) / 2
                Wi = np.round(Wo * 2) / 2
                
                if _verify(Ui.tolist(), Vi.tolist(), Wi.tolist(), n, m, p):
                    return {"U": Ui.tolist(), "V": Vi.tolist(), "W": Wi.tolist()}
                
                # Try local polish
                polished = _local_polish(Ui.tolist(), Vi.tolist(), Wi.tolist(), n, m, p)
                if polished is not None:
                    Ui2, Vi2, Wi2 = polished
                    if _verify(Ui2, Vi2, Wi2, n, m, p):
                        return {"U": Ui2, "V": Vi2, "W": Wi2}
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115  # Safety margin before 120s limit

    # Try ranks from lowest to highest
    # Based on theoretical bounds, rank should be at least max(n*m, m*p, p*n) = 18
    # but likely higher. Start from 20 and go up.
    for r in range(20, 37):
        if time.time() - start_time > time_limit:
            break
        
        # More restarts for lower ranks (harder but more valuable)
        n_restarts = 80 if r <= 22 else (50 if r <= 25 else 30)
        
        res = _try_rank_optimized(r, n_restarts, time_limit, start_time, n, m, p)
        if res is not None:
            return res

    return _schoolbook()
