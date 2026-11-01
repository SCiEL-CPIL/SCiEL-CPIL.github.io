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

def _energy_and_grad(pts):
    """Compute energy and Euclidean gradient.
    
    Energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    d(energy)/d(xi) = sum_{j!=i} (xi - xj) / |xi - xj|^2
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    energy = -math.fsum(np.log(dist[iu]))
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    # Gradient of -sum log(d) w.r.t. xi = sum_j (xi - xj)/d2_ij = xi*sum_j 1/d2_ij - sum_j xj/d2_ij
    grad = pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2
    return energy, grad

def _tangent_grad(pts):
    """Project gradient to tangent space."""
    grad = _energy_and_grad(pts)[1]
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - grad_dot_xi * pts

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

def _riemannian_gradient_descent(pts0, max_iters=2000, tol=1e-12, time_limit=10.0):
    """Fast Riemannian Gradient Descent with adaptive LR and momentum."""
    pts = pts0.copy()
    energy, _ = _energy_and_grad(pts)
    g_t = _tangent_grad(pts)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.05
    min_lr = 1e-10
    max_lr = 0.15
    momentum = 0.9
    start_time = time.time()
    stagnation = 0
    v = np.zeros_like(pts)  # momentum buffer
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        g_norm = np.linalg.norm(g_t)
        if g_norm < tol:
            break
        
        # Update momentum
        v = momentum * v - lr * g_t
        # Project velocity to tangent space
        v_dot_xi = np.sum(v * pts, axis=1, keepdims=True)
        v = v - v_dot_xi * pts
        
        dir_norm = np.linalg.norm(v)
        if dir_norm < 1e-16:
            break
        
        # Backtracking line search (fewer evaluations)
        alpha = 1.0
        improved = False
        for _ in range(12):
            new_pts = _project_to_sphere(pts + alpha * v)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy + 1e-4 * alpha * np.sum(g_t * v):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr = max(min_lr, lr * 0.5)
            momentum = max(0.5, momentum * 0.95)
            stagnation += 1
            if lr < min_lr or stagnation > 80:
                break
            continue
        
        g_t = _tangent_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            lr = min(max_lr, lr * 1.05)
            momentum = min(0.99, momentum * 1.01)
            stagnation = 0
        else:
            lr = max(min_lr, lr * 0.9)
            momentum = max(0.5, momentum * 0.98)
            stagnation += 1
    
    return best_pts, best_energy

def _riemannian_newton_polish(pts, max_iter=30, tol=1e-14, time_limit=8.0):
    """Riemannian Newton polish using approximate Hessian."""
    start_time = time.time()
    pts = pts.copy()
    energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    # Build approximate Hessian via finite differences of gradient
    # For efficiency, we use a limited-memory approach
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        g_t = _tangent_grad(pts)
        g_norm = np.linalg.norm(g_t)
        if g_norm < tol:
            break
        
        # Approximate Riemannian Hessian via central differences on tangent space
        # Use a small perturbation
        eps = 1e-6
        hessian_approx = np.zeros((N, 3, N, 3))
        
        # Only compute a few diagonal blocks for efficiency
        # Actually, let's use a simpler approach: Gauss-Newton approximation
        # The Hessian of -sum log(d_ij) has terms involving 1/d2 and 1/d4
        
        # Simpler: use conjugate gradient on the Hessian system
        # Or just do a steepest descent with very small steps
        
        # Alternative: use scipy's minimize with tight tolerance
        break  # Skip this complex approach, use L-BFGS instead
    
    return best_pts, best_energy

def _polish_lbfgs(pts, max_iter=500, gtol=1e-14, ftol=1e-16):
    """Polish using L-BFGS-B with tighter settings."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = _project_to_sphere(x_pts)
        e = _log_energy(x_pts)
        g_t = _tangent_grad(x_pts)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': ftol, 'maxls': 150, 'm': 50})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = _project_to_sphere(x_opt)
            e_opt = _log_energy(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _log_energy(pts)

def _polish_trust_constr(pts, max_iter=300):
    """Polish using trust-constr with tangent projection."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = _project_to_sphere(x_pts)
        e = _log_energy(x_pts)
        g_t = _tangent_grad(pts)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='trust-constr',
                       options={'maxiter': max_iter, 'gtol': 1e-15, 'max_nfev': 8000})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = _project_to_sphere(x_opt)
            e_opt = _log_energy(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _log_energy(pts)

def _conjugate_gradient_polish(pts, max_iters=200, tol=1e-13, time_limit=5.0):
    """Riemannian conjugate gradient for faster convergence than GD."""
    start_time = time.time()
    pts = pts.copy()
    energy, _ = _energy_and_grad(pts)
    g_t = _tangent_grad(pts)
    
    best_pts = pts.copy()
    best_energy = energy
    
    d = -g_t.copy()  # initial direction
    d_dot = np.sum(d * g_t)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        g_norm = np.linalg.norm(g_t)
        if g_norm < tol:
            break
        
        # Line search
        alpha = 1.0
        improved = False
        for _ in range(10):
            new_pts = _project_to_sphere(pts + alpha * d)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy + 1e-5 * alpha * d_dot:
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
        
        new_g_t = _tangent_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Polak-Ribiere beta
        new_d_dot = np.sum(d * new_g_t)
        beta = max(0.0, new_d_dot / max(abs(d_dot), 1e-16))
        
        d = -new_g_t + beta * d
        # Project to tangent space
        d_dot_xi = np.sum(d * pts, axis=1, keepdims=True)
        d = d - d_dot_xi * pts
        
        d_dot = np.sum(d * new_g_t)
        if d_dot >= 0:
            d = -new_g_t.copy()
            d_dot = np.sum(d * new_g_t)
        
        g_t = new_g_t
    
    return best_pts, best_energy

def _optimize_from_start(pts0, deadline):
    """Optimize from a starting configuration with multi-stage pipeline."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    # Stage 1: Riemannian GD with momentum
    if time.time() < deadline - 10.0:
        rgd_pts, rgd_energy = _riemannian_gradient_descent(best_pts, max_iters=3000, time_limit=12.0)
        if rgd_energy < best_energy:
            best_pts = rgd_pts
            best_energy = rgd_energy
    
    # Stage 2: Conjugate gradient polish
    if time.time() < deadline - 6.0:
        cg_pts, cg_energy = _conjugate_gradient_polish(best_pts, max_iters=300, time_limit=6.0)
        if cg_energy < best_energy:
            best_pts = cg_pts
            best_energy = cg_energy
    
    # Stage 3: L-BFGS Polish
    if time.time() < deadline - 3.0:
        lbfgs_pts, lbfgs_energy = _polish_lbfgs(best_pts, max_iter=1000, gtol=1e-16)
        if lbfgs_energy < best_energy:
            best_pts = lbfgs_pts
            best_energy = lbfgs_energy
    
    # Stage 4: Final high-precision L-BFGS
    if time.time() < deadline - 1.0:
        final_pts, final_energy = _polish_lbfgs(best_pts, max_iter=500, gtol=1e-17)
        if final_energy < best_energy:
            best_pts = final_pts
            best_energy = final_energy
    
    return best_pts, best_energy

def _golden_spiral_v2(n):
    """Alternative golden spiral with different phase offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + 0.3)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_icosahedron(n, rng):
    """Better icosahedron-based initialization with optimization."""
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
    
    pts = list(verts)
    
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

def construct():
    t_start = time.time()
    deadline = t_start + 118.0  # Safety margin (150s limit)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral
    initializations.append(_fibonacci_sphere(N))
    
    # 2. Alternative Golden Spiral
    initializations.append(_golden_spiral_v2(N))
    
    # 3. Icosahedron Subdivision
    initializations.append(_optimize_icosahedron(N, rng))
    
    # 4. Random Start
    initializations.append(_random_sphere(N, rng))
    
    # 5. Perturbed Fibonacci
    initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng, sigma=0.08))
    
    # 6. Perturbed Icosahedron
    initializations.append(_perturb_sphere(_optimize_icosahedron(N, rng), rng, sigma=0.05))
    
    # 7. Second random start with different seed
    rng2 = np.random.default_rng(123)
    initializations.append(_random_sphere(N, rng2))
    
    # 8. Third random start
    rng3 = np.random.default_rng(456)
    initializations.append(_random_sphere(N, rng3))
    
    best_pts = initializations[0]
    best_energy = _log_energy(best_pts)
    
    # Sort by initial energy
    init_energies = [_log_energy(p) for p in initializations]
    sorted_indices = np.argsort(init_energies)
    
    # Optimize top starts, prioritizing best initializations
    # Use more time for the best initializations
    n_optimize = min(len(sorted_indices), 5)
    for rank, idx in enumerate(sorted_indices[:n_optimize]):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate more time to earlier (better) initializations
        if rank == 0:
            alloc_time = min(remaining - 3.0, 60.0)
        elif rank == 1:
            alloc_time = min(remaining - 3.0, 40.0)
        elif rank == 2:
            alloc_time = min(remaining - 3.0, 25.0)
        else:
            alloc_time = min(remaining - 3.0, 15.0)
        
        local_deadline = time.time() + alloc_time
        init_pts = initializations[idx]
        pts, energy = _optimize_from_start(init_pts, local_deadline)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final high-precision polish if time permits
    if time.time() < deadline - 2.0:
        final_pts, final_energy = _polish_lbfgs(best_pts, max_iter=800, gtol=1e-17)
        if final_energy < best_energy:
            best_pts = final_pts
            best_energy = final_energy
    
    # One more polish with trust-constr if time permits
    if time.time() < deadline - 1.0:
        tc_pts, tc_energy = _polish_trust_constr(best_pts, max_iter=300)
        if tc_energy < best_energy:
            best_pts = tc_pts
            best_energy = tc_energy
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them slightly
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts[j] = _project_to_sphere(best_pts[j].reshape(1,3))[0]
    
    return best_pts

# EVOLVE-BLOCK-END
