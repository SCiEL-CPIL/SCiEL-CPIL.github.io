# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M_exact = 20000
    M_search = 2000  # Subsample for fast search
    
    # Precompute constants
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    
    # Keich construction initial guess
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    start_time = time.time()
    deadline = start_time + 550.0  # 50s buffer

    def _area(x, M):
        """Compute area using M quadrature points."""
        y = (np.arange(M) + 0.5) / M
        w = (1.0 - y) / n
        # L shape (M, n)
        L = x[None, :] + a[None, :] * y[:, None]
        S = np.sort(L, axis=1)
        D = np.diff(S, axis=1)
        G = np.minimum(D, w[:, None])
        union_len = w + np.sum(G, axis=1)
        return np.mean(union_len)

    def _area_grad(x, M):
        """Compute area and gradient using M quadrature points."""
        y = (np.arange(M) + 0.5) / M
        w = (1.0 - y) / n
        L = x[None, :] + a[None, :] * y[:, None]
        S = np.sort(L, axis=1)
        D = np.diff(S, axis=1)
        G = np.minimum(D, w[:, None])
        area = np.mean(w + np.sum(G, axis=1))
        
        P = np.argsort(L, axis=1)
        T = (D < w[:, None]).astype(np.float64)
        
        P_plus = P[:, 1:].ravel()
        P_minus = P[:, :-1].ravel()
        T_flat = T.ravel()
        
        grad_plus = np.bincount(P_plus, weights=T_flat, minlength=n)
        grad_minus = np.bincount(P_minus, weights=T_flat, minlength=n)
        
        grad_x = (grad_plus - grad_minus) / M
        return area, grad_x

    # Phase 1: Global Search with Simulated Annealing using M_search
    # Objective: Minimize area
    best_x = x0.copy()
    best_area_search = _area(best_x, M_search)
    
    current_x = best_x.copy()
    current_area = best_area_search
    
    # SA Parameters
    T = 0.01
    cooling = 0.995
    # We want to run as many iterations as possible
    # Estimate time per iteration:
    # _area with M=2000, n=64. 
    # 2000 * 64 * 6 ops ~ 768k ops. ~1ms?
    # Let's do 5000 iterations.
    
    np.random.seed(12345)
    
    max_iter = 10000
    for it in range(max_iter):
        if time.time() > deadline - 100:
            break
            
        # Propose move: perturb 2 random coordinates
        idx = np.random.choice(n, 2, replace=False)
        step = np.random.normal(0, 0.05, 2)
        
        new_x = current_x.copy()
        new_x[idx] += step
        
        new_area = _area(new_x, M_search)
        delta = new_area - current_area
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_x = new_x
            current_area = new_area
            if current_area < best_area_search:
                best_area_search = current_area
                best_x = current_x.copy()
        
        T *= cooling
        if T < 1e-6:
            T = 1e-6

    # Phase 2: Local Refinement with L-BFGS-B using M_exact
    # Start from the best found in Phase 1
    
    def obj_func(x):
        val, _ = _area_grad(x, M_exact)
        return val
        
    def jac_func(x):
        _, grad = _area_grad(x, M_exact)
        return grad

    best_x_final = best_x.copy()
    best_area_exact = _area(best_x_final, M_exact)

    # Run L-BFGS-B from the SA result
    try:
        res = minimize(
            obj_func, 
            best_x_final, 
            method='L-BFGS-B', 
            jac=jac_func, 
            options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() < deadline:
            if res.fun < best_area_exact:
                best_x_final = res.x
                best_area_exact = res.fun
    except Exception:
        pass

    # Phase 3: If time permits, try a few random restarts of L-BFGS-B from Keich
    if time.time() < deadline - 50:
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            # Perturb Keich slightly
            x_start = x0 + np.random.normal(0, 0.01, n)
            try:
                res = minimize(
                    obj_func, 
                    x_start, 
                    method='L-BFGS-B', 
                    jac=jac_func, 
                    options={'maxiter': 200, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_area_exact:
                    best_x_final = res.x
                    best_area_exact = res.fun
            except Exception:
                pass

    return np.array(best_x_final, dtype=np.float64)

# EVOLVE-BLOCK-END
