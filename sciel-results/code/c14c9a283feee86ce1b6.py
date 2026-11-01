"""Optimized logarithmic energy minimization for 100 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)

def _energy_and_gradient(pts):
    """Compute energy and tangent-space gradient of logarithmic energy."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    energy = -math.fsum(log_d)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2
    grad = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)
    
    # Project to tangent space
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_xi * pts
    
    return energy, grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _perturb_sphere(pts, rng, sigma=0.05):
    """Perturb points and renormalize."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def _icosahedron_subdivide(n, rng):
    """Generate points by subdividing an icosahedron."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    pts = []
    for v in verts:
        pts.append(v)
    
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1) % 3]
            edge = (min(a,b), max(a,b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        pts.append(_project_to_sphere(mid.reshape(1,3))[0])
    
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        pts.append(_project_to_sphere(center.reshape(1,3))[0])
    
    pts = np.array(pts)
    if len(pts) > n:
        indices = rng.choice(len(pts), n, replace=False)
        pts = pts[indices]
    elif len(pts) < n:
        remaining = _random_sphere(n - len(pts), rng)
        pts = np.vstack([pts, remaining])
    
    return _project_to_sphere(pts)

def _golden_spiral_v2(n, offset=0.0):
    """Alternative golden spiral with offset for diversity."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _riemannian_descent(pts, max_iter=500, tol=1e-10, deadline=None):
    """Riemannian gradient descent with backtracking line search and momentum on the sphere."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f, grad = _energy_and_gradient(pts)
    
    # Momentum buffer
    momentum = np.zeros_like(pts)
    beta = 0.9
    step_size = 1.0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Update momentum
        momentum = beta * momentum - step_size * grad
        
        # Apply momentum and project to tangent space
        momentum_dot_pts = np.sum(momentum * pts, axis=1, keepdims=True)
        momentum_tan = momentum - momentum_dot_pts * pts
        
        accepted = False
        # Backtracking line search
        for ls_iter in range(50):
            new_pts = pts + momentum_tan
            new_pts = _project_to_sphere(new_pts)
            new_f, new_grad = _energy_and_gradient(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                if new_f < f - 1e-10:
                    accepted = True
                    pts = new_pts
                    f = new_f
                    grad = new_grad
                    break
            momentum_tan *= 0.7
        
        if not accepted:
            # Reduce step size if stuck
            step_size *= 0.5
            if step_size < 1e-12:
                break
        else:
            # Gradually increase step size if making good progress
            if f < f - 1e-6:
                step_size = min(step_size * 1.1, 2.0)
    
    return pts, f

def _newton_polish(pts, deadline, max_iter=50):
    """Newton-style polish using approximate Hessian for fast local convergence."""
    pts = _project_to_sphere(pts.copy())
    f, grad = _energy_and_gradient(pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Approximate Newton step using diagonal scaling of gradient
        # This is a cheap approximation to the full Newton method
        diag_scale = 1.0 / (np.abs(grad).max(axis=0) + 1e-10)
        newton_dir = grad * diag_scale
        newton_dir_norm = np.linalg.norm(newton_dir)
        if newton_dir_norm < 1e-15:
            break
        newton_dir = newton_dir / newton_dir_norm
        
        # Line search for Newton direction
        step = 1.0
        accepted = False
        for ls_iter in range(30):
            new_pts = pts - step * newton_dir
            new_pts = _project_to_sphere(new_pts)
            new_f, new_grad = _energy_and_gradient(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                if new_f < f - 1e-10:
                    accepted = True
                    pts = new_pts
                    f = new_f
                    grad = new_grad
                    break
            step *= 0.5
        
        if not accepted:
            # Fall back to gradient direction
            step = 1.0 / (1.0 + 0.5 * grad_norm)
            accepted = False
            for ls_iter in range(20):
                new_pts = pts - step * grad
                new_pts = _project_to_sphere(new_pts)
                new_f, new_grad = _energy_and_gradient(new_pts)
                
                if np.isfinite(new_f) and new_f < f:
                    if new_f < f - 1e-10:
                        accepted = True
                        pts = new_pts
                        f = new_f
                        grad = new_grad
                        break
                step *= 0.7
            if not accepted:
                break
    
    return pts, f

def _optimize_from_start(pts0, deadline):
    """Optimize from a starting configuration with multiple stages."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_gradient(pts)
        return g.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    # Stage 1: Riemannian gradient descent with momentum (fast convergence)
    if time.time() < deadline - 30.0:
        try:
            pts_rd, e_rd = _riemannian_descent(best_pts, max_iter=1000, tol=1e-12, deadline=deadline - 30.0)
            if e_rd < best_energy:
                best_pts = pts_rd
                best_energy = e_rd
        except Exception:
            pass
    
    if time.time() > deadline - 25.0:
        return best_pts, best_energy
    
    # Stage 2: L-BFGS-B with moderate tolerance
    try:
        res = minimize(
            objective,
            best_pts.reshape(-1),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-10, 'maxfun': 1000, 'ftol': 1e-12}
        )
        if np.isfinite(res.fun) and res.fun < best_energy:
            pts_new = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    if time.time() > deadline - 15.0:
        return best_pts, best_energy
    
    # Stage 3: Riemannian gradient descent for further refinement
    if time.time() < deadline - 15.0:
        try:
            pts_rd2, e_rd2 = _riemannian_descent(best_pts, max_iter=500, tol=1e-14, deadline=deadline - 15.0)
            if e_rd2 < best_energy:
                best_pts = pts_rd2
                best_energy = e_rd2
        except Exception:
            pass
    
    if time.time() > deadline - 10.0:
        return best_pts, best_energy
    
    # Stage 4: L-BFGS-B with tighter tolerance
    try:
        x1 = best_pts.reshape(-1)
        res2 = minimize(
            objective,
            x1,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-14, 'maxfun': 2000, 'ftol': 1e-16}
        )
        if np.isfinite(res2.fun) and res2.fun < best_energy:
            pts_new = _project_to_sphere(res2.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    if time.time() > deadline - 8.0:
        return best_pts, best_energy
    
    # Stage 5: Newton polish for fast local convergence
    if time.time() < deadline - 5.0:
        try:
            pts_np, e_np = _newton_polish(best_pts, deadline - 5.0, max_iter=200)
            if e_np < best_energy:
                best_pts = pts_np
                best_energy = e_np
        except Exception:
            pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Stage 6: Final very tight L-BFGS-B
    try:
        x2 = best_pts.reshape(-1)
        res3 = minimize(
            objective,
            x2,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'gtol': 1e-16, 'maxfun': 4000, 'ftol': 1e-18}
        )
        if np.isfinite(res3.fun) and res3.fun < best_energy:
            pts_new = _project_to_sphere(res3.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    # Stage 7: Final Newton polish
    if time.time() < deadline - 2.0:
        try:
            pts_np2, e_np2 = _newton_polish(best_pts, deadline - 2.0, max_iter=300)
            if e_np2 < best_energy:
                best_pts = pts_np2
                best_energy = e_np2
        except Exception:
            pass
    
    return best_pts, best_energy

def _spectral_smooth(pts, n_iters=10, deadline=None):
    """Spectral Laplacian flow: smooth the point distribution using Delaunay neighbors."""
    pts = _project_to_sphere(pts.copy())
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise distances
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, np.inf)
        
        # Find k-nearest neighbors (k=8)
        k = 8
        neighbor_idx = np.argsort(d2, axis=1)[:, :k]
        
        # Compute weighted average of neighbors
        new_pts = np.zeros_like(pts)
        for i in range(N):
            neighbors = pts[neighbor_idx[i]]
            # Weights based on inverse distance
            dists = np.sqrt(np.maximum(d2[i, neighbor_idx[i]], 1e-15))
            weights = 1.0 / (dists + 1e-10)
            weights /= weights.sum()
            new_pts[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0)
        
        # Blend with original
        alpha = 0.3
        blended = (1.0 - alpha) * pts + alpha * new_pts
        pts = _project_to_sphere(blended)
        
        # Check if energy improved
        e_new = _log_energy(pts)
        e_old = _log_energy(_project_to_sphere((1.0 - alpha) * pts + alpha * new_pts))
        if e_new > e_old:
            break
    
    return pts

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron-based
    initializations.append(_icosahedron_subdivide(N, rng))
    
    # Alternative golden spirals with offsets
    for offset in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        initializations.append(_golden_spiral_v2(N, offset))
    
    # Random starts
    for seed_offset in range(10):
        rng_sub = np.random.default_rng(42 + seed_offset * 7)
        initializations.append(_random_sphere(N, rng_sub))
    
    # Perturbed Fibonacci
    for sigma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.3]:
        rng_sub = np.random.default_rng(1000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng_sub, sigma))
    
    # Perturbed icosahedron
    for sigma in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15]:
        rng_sub = np.random.default_rng(2000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng_sub), rng_sub, sigma))
    
    # Perturbed golden spirals
    for offset in [0.1, 0.3, 0.5, 0.7]:
        for sigma in [0.02, 0.05, 0.1]:
            rng_sub = np.random.default_rng(3000 + int(offset * 100) + int(sigma * 1000))
            initializations.append(_perturb_sphere(_golden_spiral_v2(N, offset), rng_sub, sigma))
    
    # Rotated Fibonacci
    rng_sym = np.random.default_rng(5000)
    pts_sym = _fibonacci_sphere(N)
    theta = rng_sym.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    pts_rot = _project_to_sphere(pts_sym @ rot.T)
    initializations.append(pts_rot)
    
    init_energies = [_log_energy(p) for p in initializations]
    sorted_indices = np.argsort(init_energies)
    
    best_pts = initializations[sorted_indices[0]]
    best_energy = init_energies[sorted_indices[0]]
    
    # Process top initializations
    num_to_try = min(len(sorted_indices), 15)
    for rank in range(num_to_try):
        idx = sorted_indices[rank]
        if time.time() > deadline - 30.0:
            break
        
        init_pts = initializations[idx]
        remaining = deadline - time.time()
        
        # Allocate time based on rank - give more time to top candidates
        if rank == 0:
            time_alloc = min(35.0, remaining * 0.45)
        elif rank < 3:
            time_alloc = min(25.0, remaining * 0.35)
        elif rank < 6:
            time_alloc = min(18.0, remaining * 0.25)
        elif rank < 10:
            time_alloc = min(12.0, remaining * 0.2)
        else:
            time_alloc = min(8.0, remaining * 0.15)
        
        if time_alloc < 5.0:
            break
            
        stage_deadline = time.time() + time_alloc
        pts, energy = _optimize_from_start(init_pts, stage_deadline)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Spectral smoothing pass
    if time.time() < deadline - 15.0:
        try:
            pts_smooth = _spectral_smooth(best_pts, n_iters=20, deadline=deadline - 15.0)
            e_smooth = _log_energy(pts_smooth)
            if e_smooth < best_energy:
                best_pts = pts_smooth
                best_energy = e_smooth
        except Exception:
            pass
    
    # Final polish on the best result
    if time.time() < deadline - 15.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, deadline - 5.0)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ultra-final Newton polish
    if time.time() < deadline - 3.0:
        try:
            pts_ultra, e_ultra = _newton_polish(best_pts, deadline - 3.0, max_iter=200)
            if e_ultra < best_energy:
                best_pts = pts_ultra
                best_energy = e_ultra
        except Exception:
            pass
    
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
