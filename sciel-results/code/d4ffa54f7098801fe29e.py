# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_length_1d(positions, c, N):
    """Compute the 1D union length of intervals [pos_i + c*i/N, pos_i + 1/N + c*i/N]."""
    N = len(positions)
    L = 1.0 / N
    # Interval i: [pos_i + c*i/N, pos_i + L + c*i/N]
    starts = positions + c * np.arange(N) / N
    ends = starts + L
    # Sort by start
    order = np.argsort(starts)
    s = starts[order]
    e = ends[order]
    # Merge intervals
    total = 0.0
    cur_start = s[0]
    cur_end = e[0]
    for k in range(1, N):
        if s[k] > cur_end:
            total += cur_end - cur_start
            cur_start = s[k]
            cur_end = e[k]
        else:
            if e[k] > cur_end:
                cur_end = e[k]
    total += cur_end - cur_start
    return total

def _compute_volume(x_pos, y_pos, N=8, n_samples=200):
    """Compute the exact (numerically integrated) volume V = ∫ A_x(c) * A_y(c) dc."""
    # Use Gauss-Legendre quadrature for smooth integration
    c_nodes, c_weights = np.polynomial.legendre.leggauss(n_samples)
    c_vals = 0.5 * (c_nodes + 1.0)  # map to [0,1]
    weights = 0.5 * c_weights
    
    ax = np.array([_union_length_1d(x_pos, c, N) for c in c_vals])
    ay = np.array([_union_length_1d(y_pos, c, N) for c in c_vals])
    V = np.sum(weights * ax * ay)
    return V

def _objective(params, N=8):
    """Objective: volume V. params = [x_0,...,x_7, y_0,...,y_7]"""
    x_pos = params[:N]
    y_pos = params[N:]
    return _compute_volume(x_pos, y_pos, N)

def _gradient(params, N=8, eps=1e-7):
    """Numerical gradient of the objective."""
    grad = np.zeros_like(params)
    for k in range(len(params)):
        p_plus = params.copy()
        p_plus[k] += eps
        p_minus = params.copy()
        p_minus[k] -= eps
        grad[k] = (_objective(p_plus, N) - _objective(p_minus, N)) / (2 * eps)
    return grad

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Uses L-BFGS-B optimization to minimize the exact union volume.
    """
    N = 8
    deadline = time.time() + 580  # safety margin before 630s limit
    
    # Initial guess: Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Initial parameters: x positions and y positions
    x_init = keich.copy()
    y_init = keich.copy()
    params_init = np.concatenate([x_init, y_init])
    
    best_params = params_init.copy()
    best_vol = _objective(params_init, N)
    
    # L-BFGS-B optimization
    try:
        result = minimize(
            _objective,
            params_init,
            args=(N,),
            method='L-BFGS-B',
            jac=_gradient,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50,
                     'maxfun': 10000}
        )
        if result.success or result.fun < best_vol:
            best_params = result.x
            best_vol = result.fun
    except Exception:
        pass
    
    # Try multiple random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > deadline - 120:
            break
        # Perturb initial guess
        noise = rng.normal(0, 0.05, size=2*N)
        params_try = params_init + noise
        try:
            result = minimize(
                _objective,
                params_try,
                args=(N,),
                method='L-BFGS-B',
                jac=_gradient,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50,
                         'maxfun': 5000}
            )
            if result.fun < best_vol:
                best_vol = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Construct output
    x_pos = best_params[:N]
    y_pos = best_params[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_pos[i_flat]
    base_y = y_pos[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validate: check finite and volume > 0
    if not np.all(np.isfinite(positions)):
        # Fallback to keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    # Verify box area
    X0 = np.min(x_pos)
    X1 = np.max(x_pos + (np.arange(N) + 1) / N)
    Y0 = np.min(y_pos)
    Y1 = np.max(y_pos + (np.arange(N) + 1) / N)
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
