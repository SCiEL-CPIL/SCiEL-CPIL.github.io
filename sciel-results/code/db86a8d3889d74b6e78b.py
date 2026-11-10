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
    """Riemannian conjugate gradient on the sphere."""
    n = points.shape[0]
    X = points.copy()
    e, g = _grad_and_energy(X)
    
    def _tangent(v, x):
        dot = np.sum(v * x, axis=1, keepdims=True)
        return v - dot * x
    
    g_t = _tangent(g, X)
    
    if np.linalg.norm(g_t) < tol:
        return X, e
    
    d = -g_t
    g_old_t = g_t
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        best_alpha = 1.0
        best_e = e
        best_X = X.copy()
        
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
            X_new = X + alpha * d
            X_new = _project_to_sphere(X_new)
            
            e_new = _energy(X_new)
            if e_new < best_e:
                best_e = e_new
                best_alpha = alpha
                best_X = X_new
        
        X = best_X
        e = best_e
        
        e_new, g_new = _grad_and_energy(X)
        g_new_t = _tangent(g_new, X)
        if np.linalg.norm(g_new_t) < tol:
            break
        
        beta = np.sum(g_new_t * g_new_t) / (np.sum(g_old_t * g_old_t) + 1e-30)
        d = -g_new_t + beta * d
        d = _tangent(d, X)
        
        g_old_t = g_new_t
    
    return X, e

def _lbfgs_optimize(points, max_iter=500, ftol=1e-14, gtol=1e-12, deadline=None):
    """Optimize using L-BFGS-B with projection."""
    from scipy.optimize import minimize
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(-1, 3)
        pts = _project_to_sphere(pts)
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
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
    
    return points

def _repulsion_refine(points, n_iter=500, deadline=None):
    """
    Refine using a repulsion-based approach: move each point away from its neighbors.
    This is like a discrete version of the electrostatic equilibrium.
    """
    X = points.copy()
    n = X.shape[0]
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces on each point
        forces = np.zeros_like(X)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = X[i] - X[j]
                dist_sq = np.sum(diff * diff)
                if dist_sq < 1e-15:
                    continue
                dist = np.sqrt(dist_sq)
                # Force is repulsive: proportional to 1/dist^2 in direction away
                # For log energy, the force magnitude is 1/dist
                force_mag = 1.0 / dist
                forces[i] += force_mag * diff / dist  # unit vector times magnitude
        
        # Move each point in the direction of the force (tangent to sphere)
        for i in range(n):
            # Project force to tangent space
            f_tan = forces[i] - np.sum(forces[i] * X[i]) * X[i]
            # Small step
            X[i] += 0.01 * f_tan
            X[i] = X[i] / np.linalg.norm(X[i])
    
    return X

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_fibonacci_spiral(N))
    for s in range(15):
        inits.append(_random_points(N, seed=42 + s))
    
    best_points = None
    best_energy = np.inf
    
    # Phase 1: L-BFGS on all initial configurations
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
        
        try:
            cand = _lbfgs_optimize(init_pts, max_iter=300, ftol=1e-10, gtol=1e-8, deadline=deadline)
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
    if time.time() < deadline - 30.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=500, tol=1e-12, deadline=deadline)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS polish with tighter tolerances
    if time.time() < deadline - 15.0:
        try:
            cand = _lbfgs_optimize(best_points, max_iter=1000, ftol=1e-14, gtol=1e-12, deadline=deadline)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 4: Final Riemannian CG polish
    if time.time() < deadline - 5.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=500, tol=1e-13, deadline=deadline)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS with very tight tolerances
    if time.time() < deadline - 2.0:
        try:
            cand = _lbfgs_optimize(best_points, max_iter=1000, ftol=1e-15, gtol=1e-13, deadline=deadline)
            cand_energy = _energy(cand)
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
                best_points[j] += 1e-8 * np.random.randn(3)
                best_points = _project_to_sphere(best_points)
    
    return best_points
