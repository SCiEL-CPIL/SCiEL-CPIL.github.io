import numpy as np
import time
import math

N = 150

def _energy(points):
    """Compute logarithmic energy."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu[0], iu[1]]))

def _grad_and_energy(points):
    """Compute log energy and its gradient for points on unit sphere (vectorized)."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    coeff = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    coeff[mask] = -1.0 / dist_sq[mask]
    
    row_sums = coeff.sum(axis=1)
    grad = points * row_sums[:, np.newaxis] - coeff @ points
    
    return energy, grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _riemannian_cg(points, max_iter=500, tol=1e-10, deadline=None):
    """Riemannian conjugate gradient on the sphere with Polak-Ribiere and backtracking."""
    n = points.shape[0]
    X = points.copy()
    e, g = _grad_and_energy(X)
    
    def _tangent(v, x):
        dot = np.sum(v * x, axis=1, keepdims=True)
        return v - dot * x
    
    g_t = _tangent(g, X)
    g_t_norm = np.linalg.norm(g_t)
    
    if g_t_norm < tol:
        return X, e
    
    d = -g_t
    g_old_t = g_t.copy()
    g_old_norm_sq = np.sum(g_old_t * g_old_t)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Backtracking line search with Armijo condition
        alpha = 1.0
        best_e = e
        best_X = X.copy()
        best_alpha = 1.0
        c1 = 1e-4
        directional_deriv = np.sum(g_t * d)
        
        if directional_deriv >= 0:
            # Direction is not descent, reset to steepest descent
            d = -g_t
            directional_deriv = np.sum(g_t * d)
        
        for _ls in range(30):
            X_new = X + alpha * d
            X_new = _project_to_sphere(X_new)
            
            e_new = _energy(X_new)
            if e_new < best_e + c1 * alpha * directional_deriv:
                best_e = e_new
                best_alpha = alpha
                best_X = X_new
                break
            alpha *= 0.5
        
        if best_e >= e - 1e-16:
            # No improvement, try steepest descent with small step
            d = -g_t
            directional_deriv = np.sum(g_t * d)
            if directional_deriv < 0:
                for alpha_try in [0.1, 0.01, 0.001, 0.0001]:
                    X_try = X + alpha_try * d
                    X_try = _project_to_sphere(X_try)
                    e_try = _energy(X_try)
                    if e_try < best_e:
                        best_e = e_try
                        best_alpha = alpha_try
                        best_X = X_try
                        break
            if best_e >= e:
                break
        
        X = best_X
        e = best_e
        
        e_new, g_new = _grad_and_energy(X)
        g_new_t = _tangent(g_new, X)
        g_new_norm = np.linalg.norm(g_new_t)
        g_new_norm_sq = np.sum(g_new_t * g_new_t)
        
        if g_new_norm < tol:
            break
        
        # Polak-Ribiere beta with non-negative safeguard
        if g_old_norm_sq > 1e-30:
            pr_beta = np.sum((g_new_t - g_old_t) * g_new_t) / g_old_norm_sq
            beta = max(0.0, pr_beta)
        else:
            beta = 0.0
        beta = min(beta, 10.0)
        
        d = -g_new_t + beta * d
        d = _tangent(d, X)
        
        # Check if d is a descent direction
        if np.sum(g_new_t * d) > 0:
            d = -g_new_t
            beta = 0.0
        
        g_old_t = g_new_t.copy()
        g_old_norm_sq = g_new_norm_sq
    
    return X, e

def _lbfgs_optimize(points, max_iter=500, ftol=1e-14, gtol=1e-12, deadline=None):
    """Optimize using L-BFGS-B with projection."""
    from scipy.optimize import minimize
    
    def objective_and_grad(x_flat):
        if deadline is not None and time.time() > deadline:
            return np.inf, np.zeros_like(x_flat)
        pts = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, g = _grad_and_energy(pts)
        dot = np.sum(g * pts, axis=1, keepdims=True)
        g_tangent = g - dot * pts
        return e, g_tangent.reshape(-1)
    
    try:
        result = minimize(
            objective_and_grad,
            points.reshape(-1),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 80}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
    
    return points

def _repulsion_refine(points, n_iter=500, deadline=None):
    """
    Vectorized repulsion-based refinement: move each point away from its neighbors.
    """
    X = points.copy()
    n = X.shape[0]
    step = 0.05
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Vectorized force computation
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
        dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
        np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
        
        inv_dist_sq = 1.0 / dist_sq  # (n, n)
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # forces[i] = sum_j (X[i] - X[j]) / |X[i] - X[j]|^2
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
        
        # Project to tangent space and take step
        dot = np.sum(forces * X, axis=1, keepdims=True)  # (n, 1)
        f_tan = forces - dot * X  # (n, 3)
        
        # Adaptive step size
        f_norm = np.linalg.norm(f_tan)
        if f_norm > 1e-15:
            X = X + step * f_tan / f_norm * 0.1
        else:
            X = X + step * f_tan
        
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        X = X / norms
        
        # Decay step size
        step *= 0.995
    
    return X

def _structured_grid(n):
    """Generate a structured grid of points on the sphere."""
    pts = []
    # Use a grid in spherical coordinates
    n_theta = int(np.sqrt(n * 2))
    n_phi = int(np.sqrt(n / 2))
    for i in range(n_theta):
        theta = (i + 0.5) * np.pi / n_theta
        for j in range(n_phi):
            phi = 2 * np.pi * (j + 0.5) / n_phi
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            pts.append([x, y, z])
    # Pad with random points if needed
    while len(pts) < n:
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        pts.append(v)
    return np.array(pts[:n])

def _perturb(points, rng, scale=0.1):
    """Perturb points by adding random noise and reprojecting."""
    noise = rng.randn(*points.shape) * scale
    perturbed = points + noise
    return _project_to_sphere(perturbed)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_fibonacci_spiral(N))
    inits.append(_structured_grid(N))
    for s in range(20):
        inits.append(_random_points(N, seed=100 + s))
    
    best_points = None
    best_energy = np.inf
    rng = np.random.RandomState(42)
    
    # Phase 1: L-BFGS on all initial configurations
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 70.0:
            break
        
        try:
            cand = _lbfgs_optimize(init_pts, max_iter=400, ftol=1e-10, gtol=1e-8, deadline=deadline)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = _fibonacci_spiral(N)
        best_energy = _energy(best_points)
    
    # Phase 2: Riemannian CG polish
    if time.time() < deadline - 40.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=1500, tol=1e-12, deadline=deadline)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS polish with tighter tolerances
    if time.time() < deadline - 25.0:
        try:
            cand = _lbfgs_optimize(best_points, max_iter=1500, ftol=1e-14, gtol=1e-12, deadline=deadline)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 4: Restart from perturbed best point
    if time.time() < deadline - 20.0:
        for restart in range(3):
            if time.time() > deadline - 20.0:
                break
            perturbed = _perturb(best_points, rng, scale=0.05 * (restart + 1))
            try:
                cand = _lbfgs_optimize(perturbed, max_iter=500, ftol=1e-10, gtol=1e-8, deadline=deadline)
                cand_energy = _energy(cand)
                if cand_energy < best_energy:
                    best_energy = cand_energy
                    best_points = cand.copy()
            except Exception:
                pass
    
    # Phase 5: Final Riemannian CG polish
    if time.time() < deadline - 10.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=3000, tol=1e-14, deadline=deadline)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 6: Final L-BFGS with very tight tolerances
    if time.time() < deadline - 5.0:
        try:
            cand = _lbfgs_optimize(best_points, max_iter=3000, ftol=1e-16, gtol=1e-14, deadline=deadline)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 7: Very final micro-polish with Riemannian CG
    if time.time() < deadline - 2.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=5000, tol=1e-15, deadline=deadline)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] += 1e-8 * rng.randn(3)
                best_points = _project_to_sphere(best_points)
    
    return best_points
