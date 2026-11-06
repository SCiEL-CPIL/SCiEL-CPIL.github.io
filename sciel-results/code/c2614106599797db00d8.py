"""Optimized logarithmic energy minimizer for 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return float('inf')
    return math.fsum(np.log(1.0 / d_upper).tolist())

def _log_energy_grad(pts):
    """Compute gradient of log(1/d) energy w.r.t. point positions (unconstrained)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid division by zero
    d_safe = np.where(d < 1e-15, 1e-15, d)
    # grad_i = sum_{j != i} -diff_{ij} / d_{ij}^2
    inv_d2 = np.zeros_like(d)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2 = 1.0 / (d_safe * d_safe)
    np.fill_diagonal(inv_d2, 0.0)
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset * n
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=0):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n):
    """Generate points based on icosahedron vertices, then refine with Fibonacci."""
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    # Use Fibonacci spiral as base, but perturb slightly
    base = _fibonacci_sphere(n)
    return base

def _optimize_lbfgs(points, max_time=30.0):
    """Optimize using L-BFGS-B with sphere projection via penalty or parameterization."""
    # Use a parameterization: each point is 3D, we minimize energy with projection
    # Alternative: use scipy minimize with the constraint that |x_i| = 1
    # We'll use a simple approach: minimize energy with L-BFGS-B, projecting after each step
    
    start = time.time()
    
    def objective_and_grad(flat_pts):
        pts = _project_to_sphere(flat_pts.reshape(N, 3))
        e = _log_energy(pts)
        if not np.isfinite(e):
            return e, np.zeros_like(flat_pts)
        g = _log_energy_grad(pts)
        # Project gradient to tangent space
        for i in range(N):
            g[i] -= np.dot(g[i], pts[i]) * pts[i]
        return e, g.flatten()
    
    x0 = points.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
    )
    
    if time.time() - start > max_time:
        return points, _log_energy(points)
    
    if result.success:
        best_pts = _project_to_sphere(result.x.reshape(N, 3))
        best_e = _log_energy(best_pts)
        return best_pts, best_e
    else:
        return points, _log_energy(points)

def _optimize_momentum(points, max_iters=1000, lr=0.01, momentum=0.9):
    """Momentum-based gradient descent with sphere projection."""
    pts = _project_to_sphere(points.copy())
    v = np.zeros_like(pts)
    current_energy = _log_energy(pts)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        grad = _log_energy_grad(pts)
        # Project gradient to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        v = momentum * v - lr * grad
        new_pts = _project_to_sphere(pts + v)
        new_energy = _log_energy(new_pts)
        
        if new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
        else:
            v *= 0.5
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return pts, current_energy

def _polish_high_precision(points, max_iters=2000, lr=0.001):
    """High-precision polishing with very small steps."""
    pts = _project_to_sphere(points.copy())
    current_energy = _log_energy(pts)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 2:
            break
        
        grad = _log_energy_grad(pts)
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Adaptive step size
        step = lr
        best_pts = pts
        best_e = current_energy
        
        for _ in range(20):
            new_pts = _project_to_sphere(pts - step * grad)
            new_e = _log_energy(new_pts)
            if new_e < best_e - 1e-14:
                best_pts = new_pts
                best_e = new_e
                break
            step *= 0.5
        
        if np.allclose(best_pts, pts, atol=1e-14):
            break
        
        pts = best_pts
        current_energy = best_e
    
    return pts, current_energy

def construct():
    start_time = time.time()
    
    # Generate diverse initial candidates
    candidates = []
    
    # Fibonacci spiral variants
    for offset in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        candidates.append(_fibonacci_sphere(N, offset=offset))
    
    # Random initializations
    for seed in [1, 7, 42, 123, 777, 2024, 3141, 2718]:
        candidates.append(_random_sphere(N, seed=seed))
    
    # Perturbed Fibonacci
    base = _fibonacci_sphere(N)
    rng = np.random.RandomState(999)
    for amp in [0.01, 0.05, 0.1]:
        perturbed = base + amp * rng.randn(N, 3)
        candidates.append(_project_to_sphere(perturbed))
    
    best_points = None
    best_energy = float('inf')
    
    # Phase 1: Quick screening with momentum
    for idx, init_pts in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 60:
            break
        
        optimized, energy = _optimize_momentum(init_pts, max_iters=200, lr=0.01)
        
        if energy < best_energy:
            best_energy = energy
            best_points = optimized
    
    # Phase 2: Refine best with L-BFGS-B
    if best_points is not None and time.time() - start_time < DEADLINE - 30:
        refined, energy = _optimize_lbfgs(best_points, max_time=25.0)
        if energy < best_energy:
            best_energy = energy
            best_points = refined
    
    # Phase 3: High-precision polish
    if best_points is not None and time.time() - start_time < DEADLINE - 10:
        polished, energy = _polish_high_precision(best_points, max_iters=500, lr=0.001)
        if energy < best_energy:
            best_energy = energy
            best_points = polished
    
    # Final projection and validity check
    best_points = _project_to_sphere(best_points)
    
    # Check minimum distance
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            min_dist = min(min_dist, d)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_sphere(N)
    
    return best_points
