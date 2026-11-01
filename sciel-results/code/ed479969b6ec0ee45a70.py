import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = time.time() + 145.0  # Hard wall-clock deadline with 5s safety margin

def _time_left():
    return max(0.0, DEADLINE - time.time())

def _energy_fsum(points):
    """Compute logarithmic energy using math.fsum for maximum precision."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())

def _energy_grad_vec(points):
    """Vectorized energy (np.sum, fast) and gradient on the sphere.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy (fast, for comparison during optimization)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    # Project gradient to tangent space
    g_tangent = grad - (grad * points).sum(axis=1, keepdims=True) * points
    
    return energy, g_tangent

def _project(points):
    """Project to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _riemannian_flow(points, time_budget, initial_lr=0.05, min_lr=1e-7,
                     lr_decay=0.999, patience=2000):
    """Riemannian gradient descent with backtracking line search and adaptive learning rate.
    Operates directly on the sphere manifold.
    """
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        energy, grad = _energy_grad_vec(best_points)
        
        # Check if gradient is small enough
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(20):
            # Riemannian step: move in direction -grad, then re-project
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            
            new_energy, _ = _energy_grad_vec(new_points)
            
            if new_energy < energy - 1e-10:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                break
    
    return best_points

def _polish(points, time_budget):
    """High-precision polish using L-BFGS-B with tight tolerances.
    Uses the ambient-space formulation with projection.
    """
    start_time = time.time()
    
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
        if hasattr(res, 'x'):
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points

def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)

def _random_points(n, seed):
    """Generate random points on sphere with given seed."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    return _project(points)

def _geodesic_points(n):
    """
    Generate points based on a subdivided icosahedron (geodesic dome).
    This provides a high-symmetry initial guess.
    """
    # Icosahedron vertices
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    verts = _project(verts)
    
    # If n == 12, return verts
    if n == 12:
        return verts
    
    # For n=100, we can't easily make a perfect geodesic dome.
    # We will use a hybrid: 12 icosahedron vertices + 88 points from Fibonacci spiral
    # But to keep symmetry, let's just use Fibonacci spiral as the "symmetric" guess
    # because it's the best general-purpose quasi-symmetric distribution.
    # The "geodesic" hint is best applied by using the icosahedron vertices as anchors
    # and distributing the rest.
    
    # Let's just return Fibonacci for now, as it's more robust for arbitrary N.
    return _fibonacci_spiral(n)

def construct():
    global DEADLINE
    DEADLINE = time.time() + 145.0
    
    def _time_left():
        return max(0.0, DEADLINE - time.time())
    
    best_points = None
    best_energy = float('inf')
    
    # Phase 1: Riemannian flow from Fibonacci spiral
    points_fib = _fibonacci_spiral(N)
    if _time_left() > 5.0:
        budget1 = min(_time_left() * 0.25, 20.0)
        refined = _riemannian_flow(points_fib, budget1)
        e, _ = _energy_grad_vec(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 2: Multiple random restarts with Riemannian flow
    restart_time_limit = _time_left() * 0.5
    restart_start = time.time()
    
    for seed in range(1, 1000):
        if time.time() - restart_start > restart_time_limit:
            break
        if _time_left() < 10.0:
            break
        
        random_points = _random_points(N, seed)
        
        # Quick Riemannian flow with limited budget
        budget_per = min(3.0, _time_left() * 0.1)
        if budget_per < 0.5:
            break
        
        refined = _riemannian_flow(random_points, budget_per, initial_lr=0.1, patience=500)
        e, _ = _energy_grad_vec(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 3: Polish the best with L-BFGS-B
    if _time_left() > 5.0 and best_points is not None:
        polish_budget = min(_time_left() * 0.6, 30.0)
        polished = _polish(best_points, polish_budget)
        e, _ = _energy_grad_vec(polished)
        if e < best_energy:
            best_energy = e
            best_points = polished
    
    # Final validation: ensure no points are too close
    if best_points is not None:
        best_points = _project(best_points)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < 1e-9:
                    # Perturb and re-project
                    noise = np.random.randn(3) * 1e-5
                    best_points[i] = _project(best_points[i] + noise)
        
        return best_points
    
    # Fallback
    return _fibonacci_spiral(N)
