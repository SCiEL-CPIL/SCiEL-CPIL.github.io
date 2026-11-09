"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the sphere.
    Energy = sum_{i<j} log(1 / |x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    
    We use chordal distances. The constraint is that points lie on the unit sphere.
    We'll optimize in R^3 with a penalty or use a parameterization.
    
    Actually, let's use a simpler approach: optimize on the sphere using
    gradient descent with projection, or use scipy's minimize with bounds.
    
    Better: use the fact that for points on the sphere, we can do gradient
    descent in R^3 and project back to the sphere at each step.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute all pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero: set diagonal to 1
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij)
    # We only want upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: d/dx_k of -sum_{i<j} log(|x_i - x_j|)
    # For a pair (i,j), contribution to grad at x_i is -(x_i - x_j)/|x_i - x_j|^2
    # and to x_j is (x_i - x_j)/|x_i - x_j|^2
    # So grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # Let's compute: for each i, grad_i = sum_j (x_j - x_i) / dist_ij^2
    # Note: dist_ij^2 = |x_i - x_j|^2
    
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)  # avoid div by zero
    
    # grad_i = sum_j (x_j - x_i) / dist_ij^2
    # = sum_j x_j / dist_ij^2 - x_i * sum_j 1/dist_ij^2
    # Let W_ij = 1/dist_ij^2 for i != j, W_ii = 0
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W_ij * x_j - x_i * sum_j W_ij
    grad = W @ pts - pts * np.sum(W, axis=1, keepdims=True)
    
    # Note: this is the gradient in R^3. For constrained optimization on the sphere,
    # we project the gradient to the tangent space.
    return energy, grad


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _energy(pts):
    """Compute just the energy."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent_sphere(pts, lr=0.01, max_iter=20000, tol=1e-10):
    """
    Gradient descent on the sphere with projection.
    Uses the Riemannian gradient (project Euclidean gradient to tangent space).
    """
    pts = _project_to_sphere(pts.copy())
    n = pts.shape[0]
    
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space: grad_tan = grad - (grad . x) x
        grad_tan = grad - (grad * pts).sum(axis=1, keepdims=True) * pts
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        # Adaptive learning rate
        step = lr * grad_tan
        new_pts = _project_to_sphere(pts - step)
        
        new_energy = _energy(new_pts)
        
        # Line search if energy increased
        if new_energy > energy:
            # Halve the step
            for _ in range(10):
                lr = lr * 0.5
                step = lr * grad_tan
                new_pts = _project_to_sphere(pts - step)
                new_energy = _energy(new_pts)
                if new_energy <= energy:
                    break
            else:
                # If still increasing, accept anyway with small step
                step = 0.01 * grad_tan
                new_pts = _project_to_sphere(pts - step)
                new_energy = _energy(new_pts)
        
        pts = new_pts
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = pts.copy()
    
    return best_pts, best_energy


def _basin_hopping(pts, max_restarts=50, perturb_scale=0.1, seed=42):
    """
    Basin hopping: random perturbations followed by local optimization.
    """
    rng = np.random.RandomState(seed)
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    for restart in range(max_restarts):
        # Perturb
        noise = rng.randn(*pts.shape) * perturb_scale
        perturbed = _project_to_sphere(pts + noise)
        
        # Local optimization
        opt_pts, opt_energy = _gradient_descent_sphere(perturbed, lr=0.005, max_iter=10000)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    # First, do a solid gradient descent
    opt_pts, opt_energy = _gradient_descent_sphere(pts, lr=0.01, max_iter=30000, tol=1e-12)
    if opt_energy < best_energy:
        best_energy = opt_energy
        best_pts = opt_pts.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_pts
    
    # Now try basin hopping with multiple restarts
    rng = np.random.RandomState(12345)
    
    # Try different perturbation scales
    for scale in [0.05, 0.1, 0.2, 0.3]:
        if time.time() > deadline:
            break
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            noise = rng.randn(*best_pts.shape) * scale
            perturbed = _project_to_sphere(best_pts + noise)
            
            # Local optimization with tighter tolerance
            opt_pts, opt_energy = _gradient_descent_sphere(
                perturbed, lr=0.003, max_iter=20000, tol=1e-12
            )
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
    
    # Final polish: very tight gradient descent
    if time.time() < deadline - 5:
        final_pts, final_energy = _gradient_descent_sphere(
            best_pts, lr=0.001, max_iter=50000, tol=1e-14
        )
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts.copy()
    
    # Ensure points are on the unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] = best_pts[j] + 1e-8 * rng.randn(3)
                best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
