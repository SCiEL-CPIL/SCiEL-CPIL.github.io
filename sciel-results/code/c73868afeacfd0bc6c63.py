"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

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

def _energy_and_tangent_grad(pts):
    """Compute energy and tangent-space gradient of logarithmic energy.
    
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    
    The R^3 gradient is: dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    
    Tangent gradient: project onto tangent space.
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    # Energy
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    energy = -math.fsum(log_d)
    
    # Gradient in R^3: dE/dx_i = sum_{j!=i} (x_j - x_i) / d2_ij
    # = sum_{j!=i} x_j / d2_ij - x_i * sum_{j!=i} 1/d2_ij
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)  # (n,)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T  # (n, 3)
    
    # R^3 gradient
    grad_r3 = sum_xj_inv_d2 - pts * sum_inv_d2[:, np.newaxis]
    
    # Project onto tangent space
    grad_dot_xi = np.sum(grad_r3 * pts, axis=1, keepdims=True)
    grad_tan = grad_r3 - grad_dot_xi * pts
    
    return energy, grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _retract(pts):
    """Retraction to sphere: normalize."""
    return _project_to_sphere(pts)

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

def _icosahedron_subdivide(n_target, rng):
    """Generate points via icosahedron subdivision for better initial distribution."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide until we have enough points
    current = verts.copy()
    while len(current) < n_target:
        new_points = []
        # For each edge, add midpoint if not already present
        # Simple approach: subdivide all faces
        # Use a simpler method: just add random points if we're close
        if len(current) > 50:
            # Add remaining points randomly near existing structure
            remaining = n_target - len(current)
            for _ in range(remaining):
                idx = rng.integers(len(current))
                base = current[idx]
                noise = rng.standard_normal(3) * 0.1
                new_pt = _project_to_sphere(base + noise)
                new_points.append(new_pt)
            current = np.vstack([current, np.array(new_points)])
            break
        else:
            # Subdivide by adding midpoints of edges
            midpoints = []
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    d = np.linalg.norm(current[i] - current[j])
                    if d < 0.5:  # Only add if edge is short enough
                        mid = _project_to_sphere(current[i] + current[j])
                        # Check if midpoint already exists
                        is_dup = False
                        for k in range(len(current)):
                            if np.linalg.norm(mid - current[k]) < 0.01:
                                is_dup = True
                                break
                        if not is_dup:
                            midpoints.append(mid)
            if len(midpoints) == 0:
                # Fallback: add random points
                remaining = n_target - len(current)
                for _ in range(remaining):
                    new_points.append(_random_sphere(1, rng)[0])
                current = np.vstack([current, np.array(new_points)])
                break
            current = np.vstack([current, np.array(midpoints)])
    
    # If we have too many, randomly select
    if len(current) > n_target:
        indices = rng.choice(len(current), size=n_target, replace=False)
        current = current[indices]
    
    return current[:n_target]

def _riemannian_gradient_descent(pts, max_iter=500, tol=1e-12, time_limit=None, t_start=None):
    """Riemannian gradient descent with adaptive step size."""
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _log_energy(best_pts)
    
    if t_start is None:
        t_start = time.time()
    
    step_size = 0.01
    prev_energy = best_energy
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - t_start > time_limit:
            break
        
        energy, grad_tan = _energy_and_tangent_grad(best_pts)
        
        if energy < best_energy:
            best_energy = energy
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        # Try the step
        new_pts = _project_to_sphere(best_pts - step_size * grad_tan)
        new_energy = _log_energy(new_pts)
        
        if new_energy < energy:
            # Accept step, maybe increase step size
            best_pts = new_pts
            step_size = min(step_size * 1.1, 0.1)
        else:
            # Reject step, decrease step size
            step_size *= 0.5
            if step_size < 1e-10:
                break
        
        # Convergence check
        if abs(prev_energy - energy) < tol * max(1.0, abs(energy)):
            if it > 10:
                break
        prev_energy = energy
    
    return best_pts, best_energy

def _riemannian_conjugate_gradient(pts, max_iter=300, tol=1e-13, time_limit=None, t_start=None):
    """Riemannian conjugate gradient method for faster convergence."""
    if t_start is None:
        t_start = time.time()
    
    X = _project_to_sphere(pts.copy())
    best_pts = X.copy()
    best_energy = _log_energy(X)
    
    energy, g = _energy_and_tangent_grad(X)
    d = -g  # initial descent direction
    d_norm = np.linalg.norm(d)
    if d_norm < tol:
        return X, best_energy
    
    prev_g = g.copy()
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - t_start > time_limit:
            break
        
        # Line search: backtracking
        alpha = 0.05
        for ls in range(30):
            new_pts = _project_to_sphere(X - alpha * d)
            new_energy = _log_energy(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g * d):
                break
            alpha *= 0.5
        else:
            break
        
        X = new_pts
        energy = new_energy
        
        if energy < best_energy:
            best_energy = energy
            best_pts = X.copy()
        
        new_g = _energy_and_tangent_grad(X)[1]
        
        # Check convergence
        if np.linalg.norm(new_g) < tol:
            break
        
        # Fletcher-Reeves beta
        beta = np.sum(new_g * new_g) / max(np.sum(prev_g * prev_g), 1e-30)
        beta = max(beta, 0.0)
        
        d = -new_g + beta * d
        
        # Reset direction if not descent
        if np.sum(d * new_g) > 0:
            d = -new_g
        
        prev_g = new_g
    
    return best_pts, best_energy

def _optimize_from_start(pts0, time_budget):
    """Optimize from a starting configuration using multiple methods."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    t0 = time.time()
    
    # Method 1: L-BFGS-B with corrected gradient
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g_tan = _energy_and_tangent_grad(pts)
        # Chain rule: grad_x = g_tan / |x_i|
        norms = np.linalg.norm(x.reshape(N, 3), axis=1)
        norms = np.maximum(norms, 1e-15)
        grad_x = g_tan / norms[:, np.newaxis]
        return e, grad_x.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    try:
        res = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 400, 'gtol': 1e-10, 'maxfun': 600, 'ftol': 1e-12}
        )
        pts_new = _project_to_sphere(res.x.reshape(N, 3))
        e_new = _log_energy(pts_new)
        if e_new < best_energy:
            best_pts = pts_new
            best_energy = e_new
    except Exception:
        pass
    
    # Method 2: Riemannian conjugate gradient polish
    remaining = time_budget - (time.time() - t0)
    if remaining > 3.0:
        try:
            pts_rcg, e_rcg = _riemannian_conjugate_gradient(
                best_pts, max_iter=200, tol=1e-13,
                time_limit=min(remaining * 0.4, 8.0), t_start=time.time()
            )
            if e_rcg < best_energy:
                best_pts = pts_rcg
                best_energy = e_rcg
        except Exception:
            pass
    
    # Method 3: Riemannian gradient descent fine-tune
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0:
        try:
            pts_rg, e_rg = _riemannian_gradient_descent(
                best_pts, max_iter=200, tol=1e-13,
                time_limit=min(remaining * 0.3, 5.0), t_start=time.time()
            )
            if e_rg < best_energy:
                best_pts = pts_rg
                best_energy = e_rg
        except Exception:
            pass
    
    # Method 4: Final high-precision L-BFGS polish
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0:
        try:
            x1 = best_pts.reshape(-1)
            res2 = minimize(
                lambda x: objective_and_grad(x)[0],
                x1,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 600, 'gtol': 1e-14, 'maxfun': 1000, 'ftol': 1e-15}
            )
            pts_new = _project_to_sphere(res2.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
        except Exception:
            pass
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations - prioritize quality starts
    initializations = []
    
    # Fibonacci spiral (good baseline)
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron subdivision (better initial distribution)
    initializations.append(_icosahedron_subdivide(N, rng))
    
    # Perturbed Fibonacci with different scales
    for sigma in [0.05, 0.12]:
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng, sigma))
    
    # Perturbed icosahedron
    initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng), rng, 0.05))
    
    # A few random starts for diversity
    for _ in range(3):
        initializations.append(_random_sphere(N, rng))
    
    best_pts = initializations[0]
    best_energy = _log_energy(best_pts)
    
    # Sort initializations by initial energy (best first) to allocate time wisely
    init_energies = [(init_pts, _log_energy(init_pts)) for init_pts in initializations]
    init_energies.sort(key=lambda x: x[1])
    
    # Try each initialization with adaptive time allocation
    n_inits = len(init_energies)
    for idx, (init_pts, init_e) in enumerate(init_energies):
        if time.time() > deadline - 20.0:
            break
        
        remaining = deadline - time.time()
        # Allocate more time to better initializations
        # First few get more time, later ones get less
        if idx == 0:
            time_for_this = min(remaining * 0.4, 30.0)
        elif idx < 3:
            time_for_this = min(remaining * 0.25, 20.0)
        else:
            time_for_this = min(remaining * 0.15, 12.0)
        
        if time_for_this < 3.0:
            continue
        
        pts, energy = _optimize_from_start(init_pts, time_for_this)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish on the best result - use remaining time
    if time.time() < deadline - 5.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, deadline - time.time() - 2.0)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check for duplicate points
    dots = best_pts @ best_pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    min_dist = np.sqrt(np.min(d2))
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if points are too close
        return _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
