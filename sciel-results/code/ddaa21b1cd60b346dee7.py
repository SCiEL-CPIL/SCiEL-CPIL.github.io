import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its gradient.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    
    For unit vectors:
    dE/dx_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # diff[i,j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient: grad[i] = -sum_j (x_i - x_j) / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy: -0.5 * sum_{i!=j} log(d_ij) = -0.5 * sum_{i!=j} 0.5*log(d_ij^2) = -0.25 * sum log(d_sq)
    # But we want sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij) = -0.5 * sum_{i<j} log(d_sq_ij)
    iu = np.triu_indices(N, k=1)
    d_sq_upper = dist_sq[iu]
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    return energy, grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _retract(p, v):
    """
    Riemannian retraction on the sphere: move along geodesic.
    p is on sphere, v is tangent vector (v . p = 0).
    Returns new point on sphere.
    """
    # Simple projection retraction
    p_new = p + v
    return _project_to_sphere(p_new)

def _tangent_project(p, g):
    """Project gradient to tangent space at p."""
    # g_tan = g - (g . p) * p
    gp = np.sum(g * p, axis=1, keepdims=True)
    return g - gp * p

def _riemannian_lbfgs(points, max_iter=5000, tol=1e-14, deadline=None):
    """
    Riemannian L-BFGS on the sphere.
    """
    p = points.copy()
    e, g = _energy_and_grad(p)
    best_e = e
    best_p = p.copy()
    
    # L-BFGS state
    m = 10  # memory
    s_list = []  # step vectors
    y_list = []  # gradient differences
    
    step_size = 0.1
    c1 = 1e-4  # Armijo constant
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Project gradient to tangent space
        g_tan = _tangent_project(p, g)
        
        # Check convergence
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        # For simplicity, use steepest descent direction with adaptive step
        # (full L-BFGS on manifold is complex; use gradient descent with momentum)
        d = -g_tan
        
        # Backtracking line search
        step = step_size
        accepted = False
        
        for _ in range(30):
            p_new = _retract(p, step * d)
            e_new, g_new = _energy_and_grad(p_new)
            
            if e_new < e - c1 * step * np.sum(d * g_tan):
                # Update L-BFGS memory
                s = step * d
                y = _tangent_project(p_new, g_new) - g_tan
                sy = np.sum(s * y)
                
                if sy > 1e-10:
                    s_list.append(s)
                    y_list.append(y)
                    if len(s_list) > m:
                        s_list.pop(0)
                        y_list.pop(0)
                
                p = p_new
                e = e_new
                g = g_new
                if e < best_e:
                    best_e = e
                    best_p = p.copy()
                accepted = True
                break
            step *= 0.5
        
        if not accepted:
            step_size *= 0.5
            if step_size < 1e-12:
                break
        else:
            step_size = min(step_size * 1.05, 0.5)
    
    return best_p, best_e

def _newton_polish(points, max_iter=50, tol=1e-15, deadline=None):
    """
    Newton's method on the sphere using Riemannian Hessian approximation.
    """
    p = points.copy()
    e, g = _energy_and_grad(p)
    best_e = e
    best_p = p.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_tan = _tangent_project(p, g)
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
        
        # Approximate Hessian using finite differences of gradient
        # This is expensive, so use a simpler approach: trust-region-like step
        # Use conjugate gradient on the Hessian approximation
        
        # For efficiency, use a simple Newton-like step with damping
        # Hessian approximation: H ≈ I (identity) for simplicity
        # This reduces to gradient descent with better step size
        
        # Compute a better step size using quadratic model
        # Try multiple step sizes
        best_step = 0.01
        best_e_trial = e
        
        for trial_step in [0.05, 0.01, 0.005, 0.001, 0.0005]:
            p_trial = _retract(p, -trial_step * g_tan)
            e_trial, _ = _energy_and_grad(p_trial)
            if e_trial < best_e_trial:
                best_e_trial = e_trial
                best_step = trial_step
        
        p_new = _retract(p, -best_step * g_tan)
        e_new, g_new = _energy_and_grad(p_new)
        
        if e_new < best_e:
            best_e = e_new
            best_p = p_new.copy()
        
        p = p_new
        e = e_new
        g = g_new
    
    return best_p, best_e

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _perturb_sphere(points, epsilon=0.01, seed=None):
    """Add small random perturbation to points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    noise = rng.randn(N, 3) * epsilon
    return _project_to_sphere(points + noise)

def _icosahedron_subdivide():
    """
    Generate points by subdividing an icosahedron.
    This gives a more uniform starting distribution.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces (20 faces)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide each face into smaller triangles
    # We need 100 points total, so subdivide each face into 5 triangles (4 new points per face)
    # 20 faces * 4 = 80 new points + 12 original = 92, not quite 100
    # Let's use a different approach: use Fibonacci spiral as primary
    
    # For now, just return Fibonacci spiral
    return _fibonacci_spiral()

def _optimize_from_start(points, deadline):
    """Run full optimization pipeline from a starting configuration."""
    p = points.copy()
    e, _ = _energy_and_grad(p)
    best_e = e
    best_p = p.copy()
    
    # Phase 1: Riemannian L-BFGS / gradient descent
    if time.time() < deadline - 30:
        p, e = _riemannian_lbfgs(p, max_iter=3000, tol=1e-13, deadline=deadline - 30)
        if e < best_e:
            best_e = e
            best_p = p.copy()
    
    # Phase 2: Newton polish
    if time.time() < deadline - 10:
        p, e = _newton_polish(p, max_iter=100, tol=1e-14, deadline=deadline - 10)
        if e < best_e:
            best_e = e
            best_p = p.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        p, e = _riemannian_lbfgs(p, max_iter=2000, tol=1e-15, deadline=deadline - 5)
        if e < best_e:
            best_e = e
            best_p = p.copy()
    
    return best_p, best_e

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Multiple starting points for robustness
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral())
    
    # 2. Random starts with different seeds
    for seed in [42, 123, 777, 2024]:
        starts.append(_random_points(seed))
    
    # 3. Perturbed Fibonacci
    starts.append(_perturb_sphere(_fibonacci_spiral(), epsilon=0.05, seed=99))
    
    best_points = None
    best_energy = float('inf')
    
    for i, start_pts in enumerate(starts):
        if time.time() > deadline - 20:
            break
        
        # Run optimization from this start
        p, e = _optimize_from_start(start_pts, deadline)
        
        if e < best_energy:
            best_energy = e
            best_points = p.copy()
    
    # Final validation and cleanup
    if best_points is None:
        best_points = _fibonacci_spiral()
    
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                rng = np.random.RandomState(i * 1000 + j)
                best_points[i] += rng.randn(3) * 1e-6
                best_points[i] = _project_to_sphere(best_points[i].reshape(1, 3))[0]
    
    return best_points
