"""Optimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import time
import math

N = 100

def _energy_and_grad(pts):
    """Compute log energy and gradient for points on unit sphere (chordal distances).
    
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    where d_ij = |x_i - x_j| (chordal distance)
    
    Gradient of -log(d_ij) w.r.t. x_i:
      d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j) / |x_i - x_j|^2
    """
    # Pairwise differences: (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    dist[dist < 1e-15] = 1e-15
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j!=i} -(x_i - x_j)/|x_i - x_j|^2
    # = sum_{j!=i} (x_j - x_i)/|x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so (x_j - x_i) = -diff[i,j]
    # grad_i = sum_j -diff[i,j] / dist[i,j]^2  for j != i
    #         = -sum_j diff[i,j] / dist[i,j]^2  (diagonal is 0 since diff[i,i]=0)
    
    # diff / dist^2: (N, N, 3)
    inv_dist_sq = 1.0 / (dist * dist)  # (N, N)
    inv_dist_sq[~np.isfinite(inv_dist_sq)] = 0.0
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff[i,j] = x_i - x_j
    # So grad[i] = -sum_j (x_i - x_j) * inv_dist_sq[i,j] = sum_j (x_j - x_i) * inv_dist_sq[i,j]
    # = sum_j (x_j - x_i) / dist[i,j]^2
    
    # Compute: grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    # diff shape (N,N,3), inv_dist_sq shape (N,N)
    # grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project gradient to tangent space of sphere: remove radial component
    # grad_tangent = grad - (grad . x_i) * x_i
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial * pts
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1e-15
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points as initial guess."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _sobol_sphere(n, rng=None):
    """Generate quasi-random points on sphere using Sobol-like sequence."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Use a simple low-discrepancy sequence
    points = []
    for i in range(n):
        # Golden ratio based
        frac = (i + 0.5) / n
        z = 1.0 - 2.0 * frac
        phi = 2.0 * np.pi * ((i * 0.618033988749895) % 1.0)
        r = np.sqrt(max(0, 1.0 - z * z))
        points.append([r * np.cos(phi), r * np.sin(phi), z])
    return np.array(points)


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _gradient_descent_sphere(pts, lr=0.01, max_iter=5000, tol=1e-12, deadline=None):
    """Riemannian gradient descent on sphere with backtracking line search."""
    pts = pts.copy()
    current_energy = _energy_and_grad(pts)[0]
    lr = lr
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(pts)
        current_energy = e
        
        # Check convergence
        if np.max(np.abs(g)) < tol:
            break
        
        # Gradient step with backtracking
        step_lr = lr
        new_pts = pts - step_lr * g
        new_pts = _project_to_sphere(new_pts)
        
        new_e = _energy_and_grad(new_pts)[0]
        
        # Backtracking line search
        backtracks = 0
        while (not np.isfinite(new_e) or new_e > current_energy) and backtracks < 20:
            step_lr *= 0.5
            new_pts = pts - step_lr * g
            new_pts = _project_to_sphere(new_pts)
            new_e = _energy_and_grad(new_pts)[0]
            backtracks += 1
        
        if backtracks >= 20:
            break
        
        pts = new_pts
        current_energy = new_e
        
        # Adaptive learning rate
        if new_e < current_energy * 0.999:
            lr = min(step_lr * 1.5, 0.1)
        else:
            lr = step_lr
    
    return pts, current_energy


def _newton_polish(pts, max_iter=100, deadline=None):
    """Newton-like polish using approximate Hessian via finite differences of gradient."""
    pts = pts.copy()
    current_energy = _energy_and_grad(pts)[0]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(pts)
        current_energy = e
        
        if np.max(np.abs(g)) < 1e-14:
            break
        
        # Approximate Hessian on tangent space using 5-point stencil
        # For each point, compute Hessian of energy restricted to sphere
        # This is expensive, so use a simplified approach: 
        # Use the fact that for log energy, the Hessian has a specific structure
        
        # Simplified: use conjugate gradient on the tangent space
        # Or just do a few more gradient descent steps with very small lr
        
        # For now, use small gradient steps
        lr = 1e-5
        new_pts = pts - lr * g
        new_pts = _project_to_sphere(new_pts)
        new_e = _energy_and_grad(new_pts)[0]
        
        if np.isfinite(new_e) and new_e < current_energy:
            pts = new_pts
            current_energy = new_e
        else:
            break
    
    return pts, current_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_pts = _fibonacci_sphere(N)
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Generate diverse initializations
    inits = [best_pts.copy()]
    
    # Random initializations with different seeds
    for seed in range(8):
        if time.time() > deadline - 40:
            break
        rng = np.random.default_rng(seed)
        pts = _random_sphere(N, rng)
        inits.append(pts)
    
    # Sobol-like initializations
    for seed in range(3):
        if time.time() > deadline - 40:
            break
        rng = np.random.default_rng(seed + 100)
        pts = _sobol_sphere(N, rng)
        inits.append(pts)
    
    # Perturbed Fibonacci
    for seed in range(4):
        if time.time() > deadline - 40:
            break
        rng = np.random.default_rng(seed + 200)
        for scale in [0.05, 0.1, 0.2]:
            if time.time() > deadline - 40:
                break
            pts = _fibonacci_sphere(N) + scale * rng.standard_normal((N, 3))
            pts = _project_to_sphere(pts)
            inits.append(pts)
    
    # Optimize each initialization
    for idx, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        # Multi-scale gradient descent
        pts = init.copy()
        
        # Coarse to fine optimization
        for lr in [0.05, 0.01, 0.001, 0.0001]:
            remaining_time = deadline - time.time() - 5
            if remaining_time < 2:
                break
            
            max_iter = min(2000, int(remaining_time * 50))
            if max_iter < 10:
                break
            
            pts, e = _gradient_descent_sphere(pts, lr=lr, max_iter=max_iter, 
                                               tol=1e-10, deadline=deadline - 5)
        
        # Final polish
        remaining_time = deadline - time.time() - 2
        if remaining_time > 1:
            pts, e = _newton_polish(pts, max_iter=50, deadline=deadline - 2)
        
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Final high-precision polish on best result
    remaining_time = deadline - time.time() - 1
    if remaining_time > 0.5:
        for lr in [1e-4, 1e-5, 1e-6]:
            remaining_time = deadline - time.time() - 0.5
            if remaining_time < 0.5:
                break
            max_iter = min(200, int(remaining_time * 20))
            if max_iter < 5:
                break
            best_pts, e = _gradient_descent_sphere(best_pts, lr=lr, max_iter=max_iter,
                                                   tol=1e-14, deadline=deadline - 0.5)
            if e < best_energy:
                best_energy = e
    
    return best_pts
