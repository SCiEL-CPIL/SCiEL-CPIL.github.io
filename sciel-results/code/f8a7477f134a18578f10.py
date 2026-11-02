import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _make_grid(n_theta, n_phi):
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def _params_to_points(params):
    n = len(params) // 2
    thetas = params[0:n]
    phis = params[n:2 * n]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    points = np.column_stack([x, y, z])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

def _objective(params, grid, beta):
    points = _params_to_points(params)
    dots = grid @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)

    # Soft-min over the N points for each grid point
    neg_beta_dists = -beta * dists
    max_vals = np.max(neg_beta_dists, axis=1, keepdims=True)
    shifted = neg_beta_dists - max_vals
    log_sum_exp = max_vals.squeeze() + np.log(np.sum(np.exp(shifted), axis=1))
    soft_min = -log_sum_exp / beta

    # Soft-max over grid points
    beta_sm = beta * soft_min
    max_val = np.max(beta_sm)
    log_sum_exp2 = max_val + np.log(np.sum(np.exp(beta_sm - max_val)))
    return log_sum_exp2 / beta

def construct():
    start_time = time.time()
    deadline = 140.0

    # Fibonacci spiral initial guess
    i = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_init = np.sqrt(1.0 - z_init * z_init)
    points_init = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
    thetas_init = np.arccos(np.clip(points_init[:, 2], -1.0, 1.0))
    phis_init = np.arctan2(points_init[:, 1], points_init[:, 0])
    params_init = np.concatenate([thetas_init, phis_init])

    # Coarse grid for main optimization
    grid_coarse = _make_grid(60, 120)
    beta = 15.0
    best_params = params_init.copy()
    best_val = _objective(params_init, grid_coarse, beta)

    result = minimize(_objective, params_init, args=(grid_coarse, beta),
                      method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
    if result.fun < best_val:
        best_params = result.x
        best_val = result.fun

    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        # Finer grid polish
        grid_fine = _make_grid(100, 200)
        beta2 = 30.0
        result2 = minimize(_objective, best_params, args=(grid_fine, beta2),
                          method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
        if result2.fun < best_val:
            best_params = result2.x
            best_val = result2.fun

    points = _params_to_points(best_params)
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(points)):
        points = points_init

    return points
