# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    return area

def _union_area_gradient(x, n=64, M=20000):
    """
    Compute the area and its gradient with respect to x.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    sort_idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    
    grad = np.zeros(n, dtype=np.float64)
    mask = d < w[np.newaxis, :]
    
    for j in range(n - 1):
        mask_j = mask[j, :]
        if np.any(mask_j):
            upper_idx = sort_idx[j + 1, mask_j]
            lower_idx = sort_idx[j, mask_j]
            np.add.at(grad, upper_idx, 1.0 / M)
            np.add.at(grad, lower_idx, -1.0 / M)
    
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    deadline = time.time() + 580.0  # Safety margin

    # Initial solution: Keich construction
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    best_x = x0.copy()
    best_area = _union_area(x0, n, M)
    
    def objective_grad(x):
        area, grad = _union_area_gradient(x, n, M)
        return area, grad

    # 1. Main L-BFGS optimization
    try:
        result = minimize(
            objective_grad,
            best_x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'gtol': 1e-8, 'maxfun': 1000}
        )
        if result.fun < best_area:
            best_x = result.x.copy()
            best_area = result.fun
    except Exception:
        pass

    # 2. Nelder-Mead Polish (handles non-smoothness better)
    if time.time() < deadline - 60:
        try:
            result_nm = minimize(
                lambda x: _union_area(x, n, M),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result_nm.fun < best_area:
                best_x = result_nm.x.copy()
                best_area = result_nm.fun
        except Exception:
            pass

    # 3. Random Restarts with larger perturbations
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        # Larger perturbations to escape local minima
        noise_scale = 0.5
        x_start = best_x + noise_scale * np.random.randn(n)
        
        try:
            result = minimize(
                objective_grad,
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 300}
            )
            
            if result.fun < best_area:
                best_x = result.x.copy()
                best_area = result.fun
                
                # Polish the new best with Nelder-Mead if time permits
                if time.time() < deadline - 30:
                    try:
                        result_nm = minimize(
                            lambda x: _union_area(x, n, M),
                            best_x,
                            method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7}
                        )
                        if result_nm.fun < best_area:
                            best_x = result_nm.x.copy()
                            best_area = result_nm.fun
                    except Exception:
                        pass
        except Exception:
            pass

    if not np.all(np.isfinite(best_x)):
        return x0
    if best_area <= 0:
        return x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
