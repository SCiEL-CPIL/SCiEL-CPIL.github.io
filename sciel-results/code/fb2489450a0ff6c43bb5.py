"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())


def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere (vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
    energy = -np.sum(np.log(d_upper))
    
    # grad[i] = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = np.zeros_like(d2)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2[~np.eye(n, dtype=bool)] = 1.0 / d2[~np.eye(n, dtype=bool)]
    
    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project to tangent space
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts
    
    return energy, grad


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _icosahedron_sphere(n):
    """Generate points using icosahedron-based construction."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    if n <= 12:
        return ico[:n]
    
    # Subdivide faces and project
    pts = [ico]
    current = ico.copy()
    while len(current) < n:
        new_pts = []
        for i in range(len(current)):
            for j in range(len(current)):
                if i < j:
                    mid = (current[i] + current[j]) / 2.0
                    mid /= np.linalg.norm(mid)
                    new_pts.append(mid)
        current = np.vstack([current, np.array(new_pts)])
        current = np.unique(np.round(current, 10), axis=0)
        if len(current) > 10000:
            break
    
    if len(current) >= n:
        # Select n points evenly
        indices = np.linspace(0, len(current) - 1, n, dtype=int)
        return current[indices]
    else:
        # Pad with random
        remaining = n - len(current)
        extra = _random_sphere(remaining, seed=999)
        return np.vstack([current, extra])


def _optimize_lbfgs(pts0, deadline, max_iter=500):
    """Optimize using L-BFGS-B with spherical constraint via parameterization."""
    # Use axis-angle or quaternion parameterization is complex.
    # Instead, use scipy's minimize with a penalty or just project after each step.
    # Better: use the fact that L-BFGS can handle constraints, but sphere is nonlinear.
    # Simplest robust approach: use scipy.optimize.minimize with method='L-BFGS-B' 
    # on the 3N coordinates, but that doesn't enforce the sphere.
    # 
    # Alternative: Use the tangent space gradient and a line search that projects.
    # scipy's minimize doesn't easily do custom line search with projection.
    #
    # Let's use a custom L-BFGS implementation or just use the built-in with 
    # a modified objective that includes a penalty for not being on the sphere.
    # But penalty methods can be unstable.
    #
    # Best approach for this task: Use scipy.optimize.minimize with method='BFGS' 
    # or 'L-BFGS-B' on the flat 3N vector, but project to sphere in the objective
    # and gradient. This is not standard.
    #
    # Actually, let's use a different strategy: 
    # 1. Use scipy.optimize.minimize with method='L-BFGS-B' on the 3N coordinates
    # 2. In the objective, project to sphere and compute energy
    # 3. In the gradient, compute the tangent gradient and map it back
    # This is tricky because the projection changes the gradient.
    #
    # Simpler and more robust: Use the momentum-based gradient descent we already have,
    # but improve it with better step size control and more restarts.
    #
    # Let's try a different approach: use scipy's minimize with a spherical parameterization.
    # Each point can be parameterized by (theta, phi), but this has singularities.
    # 
    # Let's stick with the projected gradient approach but use scipy's line search.
    
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Use scipy.optimize.minimize with a custom approach
    # Flatten the points
    x0 = pts.flatten()
    
    def obj_flat(x):
        pts_local = x.reshape(N, 3)
        pts_local = _project_to_sphere(pts_local)
        return _log_energy(pts_local)
    
    def grad_flat(x):
        pts_local = x.reshape(N, 3)
        pts_local = _project_to_sphere(pts_local)
        e, grad = _log_energy_grad(pts_local)
        return grad.flatten()
    
    try:
        result = minimize(
            obj_flat, x0, jac=grad_flat, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < best_energy:
            pts_result = _project_to_sphere(result.x.reshape(N, 3))
            e_result = _log_energy(pts_result)
            if e_result < best_energy:
                best_energy = e_result
                best_pts = pts_result
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_pts, best_energy
    
    return best_pts, best_energy


def _optimize_momentum(pts0, deadline, max_iter=2000):
    """Optimize using gradient descent with momentum and adaptive step size."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    v = np.zeros_like(pts)
    lr = 0.02
    momentum = 0.9
    patience = 0
    best_lr = lr
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-13:
            break
        
        if patience > 30:
            lr = max(lr * 0.5, 1e-8)
            v *= 0.5
            patience = 0
            if lr < 1e-9:
                break
        
        v = momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            if e_new < best_energy:
                best_lr = lr
                lr = min(lr * 1.02, 0.1)
        else:
            lr = max(lr * 0.7, 1e-8)
            v *= 0.3
    
    return best_pts, best_energy


def _refine_with_small_steps(pts, deadline, max_iter=1000):
    """Final refinement with very small steps."""
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    lr = 0.001
    v = np.zeros_like(pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-13:
            break
        
        v = 0.8 * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        e_new = _log_energy(pts_new)
        
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.01, 0.005)
        else:
            lr *= 0.5
            v *= 0.5
            if lr < 1e-10:
                break
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = np.inf
    
    # Multiple starting configurations
    starts = []
    
    # Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # Icosahedron-based
    starts.append(_icosahedron_sphere(N))
    
    # Several random starts with different seeds
    for seed in [42, 123, 456, 789, 101112, 202021, 303031, 404041]:
        starts.append(_random_sphere(N, seed=seed))
    
    # Optimize from each start
    for i, pts0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        # Give more time to earlier starts
        remaining = deadline - 10 - time.time()
        n_remaining = len(starts) - i
        time_per_start = remaining / max(1, n_remaining)
        local_deadline = min(deadline - 10, time.time() + time_per_start)
        
        # Try both L-BFGS and momentum
        if i < 3:  # Use L-BFGS for first few starts
            pts_opt, e_opt = _optimize_lbfgs(pts0, local_deadline)
        else:
            pts_opt, e_opt = _optimize_momentum(pts0, local_deadline)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Final refinement
    if best_pts is not None and time.time() < deadline - 5:
        pts_refined, e_refined = _refine_with_small_steps(best_pts, deadline)
        if e_refined < best_energy:
            best_energy = e_refined
            best_pts = pts_refined
    
    return best_pts
# EVOLVE-BLOCK-END
