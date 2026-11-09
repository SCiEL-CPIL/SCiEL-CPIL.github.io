"""Structurally different approach: Riemannian conjugate gradient + multi-start on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) and its gradient w.r.t. pts."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    energy = -np.sum(np.log(d_upper))
    
    grad = np.zeros_like(pts)
    diff_upper = diff[iu]
    d2_upper = dist2[iu]
    coeff = 1.0 / d2_upper
    
    ii = iu[0]
    jj = iu[1]
    np.add.at(grad, ii, -coeff[:, np.newaxis] * diff_upper)
    np.add.at(grad, jj, coeff[:, np.newaxis] * diff_upper)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    safe = np.where(norms > 1e-12, norms, 1.0)
    return pts / safe


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _random_initialization(n, rng):
    """Generate random initial points on the sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _tangent_project(v, pts):
    """Project vector v onto tangent space of sphere at pts."""
    return v - np.sum(v * pts, axis=1, keepdims=True) * pts


def _exp_map(pts, tangent_vec):
    """Exponential map: move from pts along tangent_vec on the sphere."""
    norm_tv = np.linalg.norm(tangent_vec, axis=1, keepdims=True)
    safe_norm = np.where(norm_tv > 1e-14, norm_tv, 1.0)
    cos_theta = np.cos(safe_norm)
    sin_theta = np.sin(safe_norm) / safe_norm
    # For zero norm, sin_theta/norm = 1, so result is pts
    result = cos_theta * pts + sin_theta * tangent_vec
    # Handle very small norms where sin/norm ≈ 1
    small = norm_tv < 1e-14
    result[small] = pts[small]
    return _project_to_sphere(result)


def _log_map(pts, target):
    """Logarithmic map: tangent vector from pts to target on sphere."""
    cos_theta = np.sum(pts * target, axis=1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0 + 1e-15, 1.0 - 1e-15)
    theta = np.arccos(cos_theta)
    safe_theta = np.where(theta > 1e-14, theta, 1.0)
    sin_theta = np.sin(safe_theta)
    coeff = theta / safe_theta
    result = coeff * (target - cos_theta * pts)
    small = theta < 1e-14
    result[small] = 0.0
    return result


def _riemannian_cg(pts, max_iter=200, tol=1e-10, deadline=None):
    """Riemannian conjugate gradient with line search on the sphere."""
    n = pts.shape[0]
    p = pts.copy()
    best_e, best_p = _energy_and_grad(p)
    best_p = p.copy()
    
    # Initial gradient
    e, g = _energy_and_grad(p)
    g_tan = _tangent_project(g, p)
    g_norm = np.linalg.norm(g_tan)
    
    if g_norm < tol:
        return p, best_e
    
    # Initial direction: negative gradient
    d = -g_tan.copy()
    
    # Beta coefficient (Polak-Ribiere)
    beta = 0.0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Line search: find step alpha
        alpha = 0.1
        p_new = _exp_map(p, alpha * d)
        e_new, g_new = _energy_and_grad(p_new)
        
        # Backtracking
        max_ls = 30
        for ls in range(max_ls):
            if e_new < e:
                break
            alpha *= 0.5
            p_new = _exp_map(p, alpha * d)
            e_new, g_new = _energy_and_grad(p_new)
        
        if e_new < e:
            p = p_new
            e = e_new
            g = g_new
            
            if e < best_e:
                best_e = e
                best_p = p.copy()
        
        # Check convergence
        g_tan_new = _tangent_project(g, p)
        g_norm_new = np.linalg.norm(g_tan_new)
        if g_norm_new < tol:
            break
        
        # Update direction (Polak-Ribiere)
        if it > 0:
            g_tan_old = g_tan  # previous tangent gradient
            inner = np.sum(g_tan_new * (-d))  # <g_new, -d>
            g_prev_norm_sq = np.sum(g_tan * g_tan)
            if g_prev_norm_sq > 1e-30:
                beta = max(0.0, inner / g_prev_norm_sq)
            else:
                beta = 0.0
        
        # New direction
        d = -g_tan_new + beta * d
        d = _tangent_project(d, p)
        
        # Check if direction is descent
        if np.sum(d * g_tan_new) > 0:
            d = -g_tan_new
            beta = 0.0
        
        g_tan = g_tan_new
    
    return best_p, best_e


def _lbfgs_optimize(pts, deadline):
    """Optimize using L-BFGS-B with sphere constraint via normalization."""
    n = pts.shape[0]
    
    def energy_flat(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def grad_flat(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        g_tangent = _tangent_project(g, p)
        return g_tangent.flatten()
    
    x0 = pts.flatten()
    try:
        res = minimize(energy_flat, x0, jac=grad_flat, 
                      method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-12, 'maxfun': 2000})
        p_opt = _project_to_sphere(res.x.reshape(n, 3))
        e_opt = _energy_and_grad(p_opt)[0]
        return p_opt, e_opt
    except Exception:
        return pts, _energy_and_grad(pts)[0]


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # Multiple random starts with different seeds
    for seed in [1, 7, 13, 42, 100]:
        r = np.random.default_rng(seed)
        initializations.append(_random_initialization(N, r))
    
    # Perturbed Fibonacci
    for k in range(3):
        base = _fibonacci_spiral(N)
        noise = rng.standard_normal((N, 3)) * 0.1
        perturbed = _project_to_sphere(base + noise)
        initializations.append(perturbed)
    
    # Evenly spaced on latitude bands
    for n_bands in [5, 10]:
        pts_list = []
        for band in range(n_bands):
            z = 1.0 - 2.0 * (band + 0.5) / n_bands
            r = np.sqrt(max(0, 1.0 - z * z))
            n_pts_in_band = N // n_bands + (1 if band < N % n_bands else 0)
            angles = 2.0 * np.pi * np.arange(n_pts_in_band) / n_pts_in_band + band * 0.3
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            zz = np.full_like(angles, z)
            pts_list.extend(zip(x, y, zz))
        pts_arr = np.array(pts_list[:N])
        initializations.append(_project_to_sphere(pts_arr))
    
    best_pts = initializations[0].copy()
    best_energy = _energy_and_grad(best_pts)[0]
    
    # For each initialization, run Riemannian CG then L-BFGS
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        # Riemannian conjugate gradient
        p_cg, e_cg = _riemannian_cg(init_pts, max_iter=300, tol=1e-11, deadline=deadline - 10.0)
        
        if e_cg < best_energy:
            best_energy = e_cg
            best_pts = p_cg.copy()
        
        # L-BFGS refinement
        if time.time() < deadline - 5.0:
            p_lbfgs, e_lbfgs = _lbfgs_optimize(p_cg, deadline)
            if e_lbfgs < e_cg:
                p_cg, e_cg = p_lbfgs, e_lbfgs
            
            if e_cg < best_energy:
                best_energy = e_cg
                best_pts = p_cg.copy()
        
        # Also try L-BFGS directly on the initial point if time permits
        if time.time() < deadline - 10.0 and init_idx < 4:
            p_direct, e_direct = _lbfgs_optimize(init_pts, deadline)
            if e_direct < best_energy:
                best_energy = e_direct
                best_pts = p_direct.copy()
    
    # Final refinement on best
    if time.time() < deadline - 3.0:
        p_final, e_final = _lbfgs_optimize(best_pts, deadline)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = p_final
    
    # One more Riemannian CG pass
    if time.time() < deadline - 2.0:
        p_final2, e_final2 = _riemannian_cg(best_pts, max_iter=200, tol=1e-12, deadline=deadline - 1.0)
        if e_final2 < best_energy:
            best_energy = e_final2
            best_pts = p_final2
    
    # Validate
    result = _project_to_sphere(best_pts)
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(result)):
        result = _fibonacci_spiral(N)
    
    return result

