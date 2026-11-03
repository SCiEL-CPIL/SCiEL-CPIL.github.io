"""Optimize logarithmic energy of 150 points on the unit sphere using Riemannian optimization."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient on the sphere.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|xi-xj|
    
    Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # inv_dist_sq[i,j] = 1/|xi-xj|^2, with diagonal = 0
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Gradient: grad[i] = sum_j (pts[i] - pts[j]) / |pts[i]-pts[j]|^2
    # = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    # Filter out any very small distances (shouldn't happen)
    mask = d > 1e-15
    d = d[mask]
    if len(d) == 0:
        energy = 0.0
    else:
        energy = -np.sum(np.log(d))
    
    return energy, grad


def _energy_only(pts):
    """Compute just the energy (for speed during optimization)."""
    energy, _ = _energy_and_grad(pts)
    return energy


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    pts = rng.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _icosahedron_based(n, seed=None):
    """Generate points based on icosahedron subdivision for better initial spread."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n <= 12:
        return verts[:n]
    
    # Use Fibonacci as fallback for simplicity, but with better initial distribution
    return _fibonacci_sphere(n, seed=seed)


def _riemannian_lbfgs(pts, time_limit=30.0, tol=1e-10):
    """Riemannian L-BFGS optimization using scipy's minimize with custom constraints.
    
    We use a retraction-based approach: optimize in R^3 with projection to sphere.
    """
    start_time = time.time()
    n = len(pts)
    
    # Flatten points for scipy
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 10000}
        )
        best_pts = _project_to_sphere(result.x.reshape(n, 3))
    except Exception:
        best_pts = _project_to_sphere(pts.copy())
    
    return best_pts


def _adam_optimize(pts, lr=0.01, max_iter=5000, time_limit=30.0, tol=1e-10):
    """Adam optimizer on the sphere with adaptive learning rate."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    
    m = np.zeros_like(pts)  # First moment
    v = np.zeros_like(pts)  # Second moment
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    prev_energy = _energy_only(pts)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent ** 2)
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        
        pts = pts - lr * m_hat / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)
        
        new_energy = _energy_only(pts)
        
        if abs(prev_energy - new_energy) < tol:
            break
        prev_energy = new_energy
    
    return pts


def _simulated_annealing(pts, time_limit=20.0):
    """Simulated annealing for escaping local minima."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    
    T_start = 0.1
    T_end = 1e-6
    prev_energy = _energy_only(pts)
    best_pts = pts.copy()
    best_energy = prev_energy
    
    start_time = time.time()
    it = 0
    total_iters = 5000
    
    for it in range(total_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (it / total_iters)
        
        # Generate perturbation
        rng = np.random.RandomState(it)
        perturb = rng.randn(n, 3) * 0.05
        new_pts = _project_to_sphere(pts + perturb)
        
        new_energy = _energy_only(new_pts)
        delta = new_energy - prev_energy
        
        # Accept or reject
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            prev_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
    
    return best_pts


def _final_polish(pts, time_limit=15.0):
    """Final high-precision polish with very tight tolerance."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    
    start_time = time.time()
    prev_energy = _energy_only(pts)
    
    # Use L-BFGS with very tight tolerance
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000}
        )
        polished = _project_to_sphere(result.x.reshape(n, 3))
    except Exception:
        polished = pts
    
    return polished


def _velocity_verlet(pts, time_limit=20.0):
    """Velocity Verlet integration on the sphere for optimization."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    vel = np.zeros_like(pts)
    
    dt = 0.01
    prev_energy = _energy_only(pts)
    start_time = time.time()
    
    for it in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        _, grad = _energy_and_grad(pts)
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Update velocity
        vel = 0.99 * vel - dt * grad_tangent
        
        # Update position
        pts = pts + dt * vel
        pts = _project_to_sphere(pts)
        
        new_energy = _energy_only(pts)
        if abs(prev_energy - new_energy) < 1e-12:
            break
        prev_energy = new_energy
    
    return pts


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Strategy: Multiple diverse initializations, each optimized with different methods,
    # then pick the best and do final polish.
    
    best_pts = None
    best_energy = float('inf')
    
    def try_candidate(pts, label=""):
        nonlocal best_pts, best_energy
        if time.time() - start_time > deadline - 5.0:
            return
        pts = _project_to_sphere(pts)
        e = _energy_only(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Phase 1: Generate diverse initial configurations
    candidates = []
    
    # Fibonacci spiral
    candidates.append(_fibonacci_sphere(N))
    
    # Random with different seeds
    for seed in [42, 123, 456, 789, 1011]:
        candidates.append(_random_sphere(N, seed=seed))
    
    # Perturbed Fibonacci
    rng = np.random.RandomState(999)
    for _ in range(3):
        pert = rng.randn(N, 3) * 0.1
        candidates.append(_project_to_sphere(_fibonacci_sphere(N) + pert))
    
    # Phase 2: Optimize each candidate with time budget
    # Allocate time proportionally
    n_candidates = len(candidates)
    base_time_per_candidate = min(15.0, (deadline - 20.0) / n_candidates)
    
    for i, cand in enumerate(candidates):
        if time.time() - start_time > deadline - 15.0:
            break
        
        remaining = deadline - 15.0 - (time.time() - start_time)
        if remaining < 2.0:
            break
        
        t_alloc = min(base_time_per_candidate, remaining)
        
        # Use L-BFGS for most, Adam for some
        if i % 3 == 0:
            opt_pts = _riemannian_lbfgs(cand, time_limit=t_alloc)
        elif i % 3 == 1:
            opt_pts = _adam_optimize(cand, lr=0.01, time_limit=t_alloc)
        else:
            opt_pts = _riemannian_lbfgs(cand, time_limit=t_alloc * 0.7)
            # If time remains, do a quick SA
            if time.time() - start_time < deadline - 10.0:
                opt_pts = _simulated_annealing(opt_pts, time_limit=5.0)
        
        try_candidate(opt_pts, f"candidate_{i}")
    
    # Phase 3: Final polish on the best configuration
    if best_pts is not None and time.time() - start_time < deadline - 10.0:
        polish_time = min(15.0, deadline - 10.0 - (time.time() - start_time))
        polished = _final_polish(best_pts, time_limit=polish_time)
        try_candidate(polished, "polished")
    
    # Phase 4: One more round of local search around the best
    if best_pts is not None and time.time() - start_time < deadline - 5.0:
        remaining = deadline - 5.0 - (time.time() - start_time)
        if remaining > 5.0:
            # Try a few perturbations of the best
            for seed in [1, 2, 3]:
                if time.time() - start_time > deadline - 3.0:
                    break
                rng = np.random.RandomState(seed)
                perturb = rng.randn(N, 3) * 0.02
                perturbed = _project_to_sphere(best_pts + perturb)
                opt_pts = _riemannian_lbfgs(perturbed, time_limit=min(5.0, remaining - 3.0))
                try_candidate(opt_pts, f"perturb_{seed}")
    
    # Final validation
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    best_pts = _project_to_sphere(best_pts)
    
    # Check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
        best_pts = _project_to_sphere(best_pts)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
