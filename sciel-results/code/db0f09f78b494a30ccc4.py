"""Rewritten program: Riesz/Logarithmic energy minimisation via Multistart Riemannian Gradient Descent."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math


def _log_energy(pts):
    """Compute sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)."""
    # Vectorized distance calculation
    # pts shape (N, 3)
    # dists[i, j] = ||pts[i] - pts[j]||
    # Use broadcasting for efficiency, or scipy if available, but stdlib/numpy is preferred for "new" feel? 
    # NumPy is allowed.
    dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    d = np.maximum(d, 1e-15)
    # We need sum over i < j, so divide by 2
    return -0.5 * np.sum(np.log(d))


def _energy_and_grad(pts):
    """Return energy and Riemannian gradient for points on the unit sphere."""
    # Compute pairwise distances squared
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid div by zero for self
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # Gradient of E = -sum_{i<j} log(dist_ij)
    # dE/dx_i = -sum_{j != i} (x_i - x_j) / dist_ij^2
    # Note: dist_ij^2 = ||x_i - x_j||^2
    # term_ij = (x_i - x_j) / dist_ij^2
    # grad_i = - sum_{j != i} term_ij
    
    # We compute the sum over j for each i.
    # inv_dist_sq = 1 / dist_sq
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = - sum_j (pts[i] - pts[j]) * inv_dist_sq[i, j]
    # diff is (x_i - x_j)
    # We want sum_j diff[i, j] * inv_dist_sq[i, j]
    # Multiply diff by inv_dist_sq along the last axis? No, inv_dist_sq is (N, N), diff is (N, N, 3).
    # We need to multiply diff[i, j, :] by inv_dist_sq[i, j].
    # reshape inv_dist_sq to (N, N, 1)
    weights = inv_dist_sq[:, :, np.newaxis]
    weighted_diff = diff * weights
    grad = -np.sum(weighted_diff, axis=1)
    
    # Project gradient to tangent space
    # grad_proj[i] = grad[i] - (grad[i] . pts[i]) * pts[i]
    grad_dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - grad_dot * pts
    
    # Energy calculation (re-use dist_sq for efficiency if possible, but _log_energy is separate)
    # For consistency, let's just call _log_energy or compute it here.
    # dist_sq has diagonal 1.0, so log(1)=0, sum over all i,j is 2 * sum_{i<j}
    # d = sqrt(dist_sq)
    # E = -0.5 * sum_{i!=j} log(sqrt(dist_sq)) = -0.25 * sum_{i!=j} log(dist_sq)
    # Since diagonal is 0 in sum, we can just sum log(dist_sq) where dist_sq > 0.
    # dist_sq diagonal is 1, log(1)=0.
    energy = -0.25 * np.sum(np.log(dist_sq))
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(N, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    return _project_to_sphere(pts)


def _riemannian_gd(pts, max_iters=2000, lr_init=0.01, lr_decay=0.999, momentum=0.9):
    """
    Perform Riemannian Gradient Descent with Momentum.
    """
    pts = _project_to_sphere(pts.copy())
    energy, grad = _energy_and_grad(pts)
    v = np.zeros_like(pts)
    lr = lr_init
    best_energy = energy
    best_pts = pts.copy()
    
    for k in range(max_iters):
        # Momentum update
        v = momentum * v - lr * grad
        # Retraction: project back to sphere
        new_pts = _project_to_sphere(pts + v)
        
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            grad = new_grad
            # Decay LR slightly
            lr *= lr_decay
        else:
            # Step was bad, reduce LR and retry with smaller step or just break if stuck?
            # Simple strategy: reduce LR significantly and try again from current pts
            lr *= 0.5
            if lr < 1e-8:
                break
            # Don't update pts, v should probably be reset or damped
            v *= 0.5
            
    return best_pts, best_energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    best_pts = None
    best_energy = float('inf')
    
    # Multiple starting points
    starts = []
    starts.append(_fibonacci_spiral(N))
    starts.append(_random_sphere(N, seed=1))
    starts.append(_random_sphere(N, seed=42))
    starts.append(_random_sphere(N, seed=123))
    
    # One "structured" random: 4 random clusters
    rng = np.random.default_rng(99)
    centers = rng.normal(size=(4, 3))
    centers = _project_to_sphere(centers)
    pts = np.zeros((N, 3))
    for i in range(N):
        c = centers[i % 4]
        # Add small noise
        noise = rng.normal(0, 0.5, 3)
        pts[i] = _project_to_sphere(c + noise)
    starts.append(pts)
    
    for i, start in enumerate(starts):
        # Run Riemannian GD
        # Use different LR/momentum for diversity?
        # Standard params first
        pts, energy = _riemannian_gd(start, max_iters=1000, lr_init=0.005, momentum=0.9)
        
        # Polish with a second, finer GD run
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Refine
        if energy < best_energy:
            pts_refined, energy_refined = _riemannian_gd(pts, max_iters=2000, lr_init=0.001, momentum=0.95)
            if energy_refined < best_energy:
                best_energy = energy_refined
                best_pts = pts_refined.copy()

    # Final Polish with L-BFGS-B? The directive says "do not reuse current designs". 
    # L-BFGS-B was the core of the previous design. 
    # However, using a polish step is standard. 
    # To be safe and strictly adhere to "new family", I will stick to GD 
    # but add a very fine-grained GD step at the end.
    
    # One more very fine GD step
    if best_pts is not None:
        pts_final, energy_final = _riemannian_gd(best_pts, max_iters=5000, lr_init=0.0001, momentum=0.99)
        if energy_final < best_energy:
            best_pts = pts_final
            best_energy = energy_final

    # Validation
    best_pts = _project_to_sphere(best_pts)
    dists = np.sqrt(np.sum((best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :])**2, axis=2))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        rng = np.random.default_rng(42)
        best_pts = best_pts + rng.normal(0, 1e-6, best_pts.shape)
        best_pts = _project_to_sphere(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
