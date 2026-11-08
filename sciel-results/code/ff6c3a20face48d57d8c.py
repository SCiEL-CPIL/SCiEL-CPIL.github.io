import numpy as np
from scipy.optimize import minimize
import time

# Precompute matrices for the rational function representation
# c = [c0, c4, c8, c12]
# sigma_p^2 = c^T P c / (c^T c) where P = diag(1, 5, 9, 13)
# sigma_x^2 = c^T X c / (c^T c) where X has entries K_nm
# Objective = (c^T P c)(c^T X c) / (c^T c)^2

# Degrees
_degrees = np.array([0, 4, 8, 12])

# P matrix: diag(deg+1)
_P = np.diag(_degrees + 1.0)

# X matrix: K_nm
_X = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        n, m = _degrees[i], _degrees[j]
        if n == m:
            _X[i, j] = n + 0.5
        elif abs(n - m) == 2:
            if n > m:
                _X[i, j] = 0.5 * np.sqrt(n * (n - 1))
            else:
                _X[i, j] = 0.5 * np.sqrt((n + 1) * (n + 2))

# Constraint: c0 + 12*c4 + 1680*c8 + 3991680*c12 = 0
# c12 = -(c0 + 12*c4 + 1680*c8) / 3991680
# We parameterize by (c0, c4, c8) and compute c12.

def _c12_from(c0, c4, c8):
    return -(c0 + 12.0 * c4 + 1680.0 * c8) / 3991680.0

def _full_coeffs(params):
    """params = [c0, c4, c8], returns [c0, c4, c8, c12]"""
    c0, c4, c8 = params
    c12 = _c12_from(c0, c4, c8)
    return np.array([c0, c4, c8, c12])

def _objective_and_grad(params):
    """
    Returns (objective_value, gradient) for params = [c0, c4, c8].
    Objective = (c^T P c)(c^T X c) / (c^T c)^2 where c = [c0, c4, c8, c12]
    """
    c = _full_coeffs(params)
    
    # Compute quadratic forms
    N = c @ c  # c^T c
    if N < 1e-30:
        return np.inf, np.zeros(3)
    
    p_form = c @ _P @ c  # c^T P c
    x_form = c @ _X @ c  # c^T X c
    
    obj = (p_form * x_form) / (N * N)
    
    # Gradient via chain rule
    dN_dc = 2.0 * c
    dp_dc = 2.0 * (_P @ c)
    dx_dc = 2.0 * (_X @ c)
    
    df_dN = -2.0 * p_form * x_form / (N ** 3)
    df_dp = x_form / (N ** 2)
    df_dx = p_form / (N ** 2)
    
    grad_c = df_dN * dN_dc + df_dp * dp_dc + df_dx * dx_dc
    
    # Gradient w.r.t. params = [c0, c4, c8]
    d_c12_dc0 = -1.0 / 3991680.0
    d_c12_dc4 = -12.0 / 3991680.0
    d_c12_dc8 = -1680.0 / 3991680.0
    
    grad_params = np.array([
        grad_c[0] + grad_c[3] * d_c12_dc0,
        grad_c[1] + grad_c[3] * d_c12_dc4,
        grad_c[2] + grad_c[3] * d_c12_dc8
    ])
    
    return obj, grad_params

def _obj_2d(c4, c8):
    """Objective with c0=1, params (c4, c8)."""
    val, _ = _objective_and_grad(np.array([1.0, c4, c8]))
    return val

def _grad_2d(c4, c8):
    """Gradient w.r.t. (c4, c8) with c0=1."""
    _, grad = _objective_and_grad(np.array([1.0, c4, c8]))
    return np.array([grad[1], grad[2]])

def _hessian_2d(c4, c8):
    """Numerical Hessian w.r.t. (c4, c8) with c0=1."""
    eps = 1e-7
    H = np.zeros((2, 2))
    f0 = _obj_2d(c4, c8)
    for i in range(2):
        for j in range(i, 2):
            e_i = np.zeros(2); e_i[i] = eps
            e_j = np.zeros(2); e_j[j] = eps
            p = np.array([c4, c8])
            H[i, j] = (_obj_2d(*(p + e_i + e_j)) - _obj_2d(*(p + e_i - e_j))
                       - _obj_2d(*(p - e_i + e_j)) + _obj_2d(*(p - e_i - e_j))) / (4 * eps * eps)
            H[j, i] = H[i, j]
    return H

def _newton_refine(x0, max_iter=100, tol=1e-14):
    """Newton's method in 2D. x0 = (c4, c8)."""
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        g = _grad_2d(*x)
        if np.linalg.norm(g) < tol:
            break
        H = _hessian_2d(*x)
        # Damped Newton step
        step = np.linalg.solve(H + 1e-12 * np.eye(2), -g)
        # Line search
        alpha = 1.0
        f_cur = _obj_2d(*x)
        for _ls in range(50):
            x_trial = x + alpha * step
            f_trial = _obj_2d(*x_trial)
            if f_trial < f_cur:
                break
            alpha *= 0.5
        else:
            break
        x = x + alpha * step
        if alpha < 1e-15:
            break
    return x, _obj_2d(*x)

def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin before 210s kill
    
    # Phase 1: Dense global grid search over (c4, c8) space
    # The objective is homogeneous of degree 0, so we fix c0=1.
    # Grid covers a wide range to find the global basin.
    
    # Use log-spaced and linear grids to cover multiple scales
    c4_vals = np.concatenate([
        np.linspace(-0.5, 0.5, 200),
        np.linspace(-2.0, 2.0, 200),
        np.linspace(-10.0, 10.0, 100),
        np.logspace(-4, 1, 50),
        -np.logspace(-4, 1, 50),
    ])
    c8_vals = np.concatenate([
        np.linspace(-0.05, 0.05, 200),
        np.linspace(-0.5, 0.5, 200),
        np.linspace(-2.0, 2.0, 100),
        np.logspace(-6, 0, 50),
        -np.logspace(-6, 0, 50),
    ])
    
    best_val = np.inf
    best_params = np.array([1.0, -0.035, -0.00027])
    
    # Evaluate grid points (vectorized where possible)
    # To keep memory manageable, process in chunks
    c4_unique = np.unique(c4_vals)
    c8_unique = np.unique(c8_vals)
    
    # Evaluate all combinations in batches
    batch_size = 50000
    n_c4 = len(c4_unique)
    n_c8 = len(c8_unique)
    
    # Use meshgrid but process in rows to limit memory
    for i_start in range(0, n_c4, 50):
        if time.time() - start_time > deadline:
            break
        i_end = min(i_start + 50, n_c4)
        c4_batch = c4_unique[i_start:i_end]
        
        for j_start in range(0, n_c8, 50):
            if time.time() - start_time > deadline:
                break
            j_end = min(j_start + 50, n_c8)
            c8_batch = c8_unique[j_start:j_end]
            
            # Vectorized evaluation
            C4, C8 = np.meshgrid(c4_batch, c8_batch, indexing='ij')
            n_pts = C4.size
            
            # Process in sub-batches to limit memory
            for sb in range(0, n_pts, 10000):
                if time.time() - start_time > deadline:
                    break
                sb_end = min(sb + 10000, n_pts)
                c4_sb = C4.flatten()[sb:sb_end]
                c8_sb = C8.flatten()[sb:sb_end]
                
                # Vectorized objective
                vals = np.empty(len(c4_sb))
                for k in range(len(c4_sb)):
                    c0 = 1.0
                    c4 = c4_sb[k]
                    c8 = c8_sb[k]
                    c12 = _c12_from(c0, c4, c8)
                    c = np.array([c0, c4, c8, c12])
                    N = c @ c
                    if N < 1e-30:
                        vals[k] = np.inf
                    else:
                        p_form = c @ _P @ c
                        x_form = c @ _X @ c
                        vals[k] = (p_form * x_form) / (N * N)
                
                min_idx = np.argmin(vals)
                if vals[min_idx] < best_val:
                    best_val = vals[min_idx]
                    best_params = np.array([1.0, c4_sb[min_idx], c8_sb[min_idx]])
    
    # Phase 2: Newton refinement from the best grid point and its neighbors
    if time.time() - start_time < deadline:
        try:
            x_refined, val_refined = _newton_refine(best_params[1:])
            if val_refined < best_val:
                best_val = val_refined
                best_params = np.array([1.0, x_refined[0], x_refined[1]])
        except Exception:
            pass
    
    # Phase 3: Additional Newton starts from top grid candidates
    # Re-evaluate a finer grid around the best point found
    if time.time() - start_time < deadline:
        try:
            c4_best = best_params[1]
            c8_best = best_params[2]
            
            # Fine local grid around best
            local_c4 = c4_best + np.linspace(-0.1, 0.1, 30)
            local_c8 = c8_best + np.linspace(-0.01, 0.01, 30)
            
            local_best_val = best_val
            local_best_x = best_params[1:]
            
            for c4_l in local_c4:
                if time.time() - start_time > deadline:
                    break
                for c8_l in local_c8:
                    v = _obj_2d(c4_l, c8_l)
                    if v < local_best_val:
                        local_best_val = v
                        local_best_x = np.array([c4_l, c8_l])
            
            if local_best_val < best_val:
                best_val = local_best_val
                best_params = np.array([1.0, local_best_x[0], local_best_x[1]])
            
            # Newton from the local best
            x_refined, val_refined = _newton_refine(local_best_x)
            if val_refined < best_val:
                best_val = val_refined
                best_params = np.array([1.0, x_refined[0], x_refined[1]])
        except Exception:
            pass
    
    # Phase 4: L-BFGS-B polish from the best point
    if time.time() - start_time < deadline:
        try:
            res = minimize(
                lambda p: _obj_2d(*p),
                best_params[1:],
                jac=lambda p: _grad_2d(*p),
                method='L-BFGS-B',
                bounds=[(-10.0, 10.0), (-2.0, 2.0)],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = np.array([1.0, res.x[0], res.x[1]])
        except Exception:
            pass
    
    # Phase 5: Final Newton polish
    if time.time() - start_time < deadline:
        try:
            x_final, val_final = _newton_refine(best_params[1:], max_iter=200, tol=1e-15)
            if val_final < best_val:
                best_val = val_final
                best_params = np.array([1.0, x_final[0], x_final[1]])
        except Exception:
            pass
    
    return best_params
