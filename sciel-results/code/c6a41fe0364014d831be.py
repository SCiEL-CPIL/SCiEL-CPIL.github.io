"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.spatial import cKDTree

N = 150

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_fast(pts):
    """Vectorized energy computation (less accurate but fast for comparisons)."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Mask diagonal
    np.fill_diagonal(dists, 1.0)
    # Take upper triangle
    iu = np.triu_indices(n, 1)
    d = dists[iu]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))


def _log_energy_grad(pts):
    """
    Compute gradient of E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            grad[i] += d / (dist * dist)
    return grad


def _project_to_sphere(pts):
    """Normalize each point to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_space(pts):
    """
    For each point on the sphere, define an orthonormal basis for the tangent space.
    Returns T[i] of shape (3, 2) such that T[i] @ v gives the 3D displacement for tangent vector v.
    """
    n = len(pts)
    T = np.zeros((n, 3, 2))
    for i in range(n):
        p = pts[i]
        # Find a vector not parallel to p
        if abs(p[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = ref - np.dot(ref, p) * p
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(p, e1)
        T[i, :, 0] = e1
        T[i, :, 1] = e2
    return T


def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_icosahedron(n_target):
    """
    Generate points via icosahedron subdivision, then pick n_target points
    by selecting a subset (or just use fibonacci as fallback if subdivision doesn't match).
    For simplicity, we use fibonacci as the base and rely on optimization.
    """
    # Actually, let's just use fibonacci sphere as the initial configuration.
    # The geodesic dome idea is complex to implement correctly for arbitrary N.
    # We'll use fibonacci and let the optimizer do the work.
    return _fibonacci_sphere(n_target)


def _local_refine(pts, max_iters=200, tol=1e-12):
    """
    Local refinement using Newton's method in tangent space.
    This minimizes the log energy with the constraint that points stay on the sphere.
    """
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    current_energy = _log_energy_fast(pts)
    
    for it in range(max_iters):
        # Compute gradient
        grad = _log_energy_grad(pts)
        
        # Project gradient to tangent space
        T = _tangent_space(pts)
        # grad_tangent[i] = T[i].T @ grad[i]
        grad_tan = np.einsum('nij,nj->ni', T, grad)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        # Compute Hessian in tangent space (approximate)
        # For log energy, the Hessian in tangent space can be computed, but it's expensive.
        # Use a simple gradient descent with adaptive step size instead.
        
        # Adaptive gradient descent
        lr = 0.01 / (1 + it * 0.1)
        step = lr * grad_tan
        
        # Update in tangent space
        pts_new = pts.copy()
        for i in range(n):
            pts_new[i] = pts[i] + T[i] @ step[i]
        
        # Project back to sphere
        pts_new = _project_to_sphere(pts_new)
        
        # Check energy improvement
        new_energy = _log_energy_fast(pts_new)
        if new_energy < current_energy:
            pts = pts_new
            current_energy = new_energy
        else:
            # Reduce step size
            lr *= 0.5
            step = lr * grad_tan
            pts_new = pts.copy()
            for i in range(n):
                pts_new[i] = pts[i] + T[i] @ step[i]
            pts_new = _project_to_sphere(pts_new)
            new_energy = _log_energy_fast(pts_new)
            if new_energy < current_energy:
                pts = pts_new
                current_energy = new_energy
            else:
                break
    
    return pts, current_energy


def _simulated_annealing(pts, max_iters=5000, T_start=0.1, T_end=1e-6):
    """
    Simulated annealing to escape local minima.
    """
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    current_energy = _log_energy_fast(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    for it in range(max_iters):
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (it / max_iters)
        
        # Generate random perturbation
        idx = np.random.randint(n)
        # Random direction in tangent space
        p = pts[idx]
        if abs(p[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = ref - np.dot(ref, p) * p
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(p, e1)
        
        # Random small displacement
        scale = 0.05 * (1.0 + 0.5 * math.sin(it * 0.1))
        delta = np.random.randn(2) * scale
        pts_new = pts.copy()
        pts_new[idx] = pts[idx] + e1 * delta[0] + e2 * delta[1]
        pts_new = _project_to_sphere(pts_new)
        
        new_energy = _log_energy_fast(pts_new)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            pts = pts_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci sphere
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _log_energy_fast(pts)
    
    # Phase 1: Local refinement from the initial configuration
    if time.time() - start_time < deadline - 60:
        pts, energy = _local_refine(pts, max_iters=100)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Simulated annealing to explore landscape
    if time.time() - start_time < deadline - 30:
        pts, energy = _simulated_annealing(best_pts, max_iters=3000, T_start=0.05, T_end=1e-5)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 3: Final local refinement
    if time.time() - start_time < deadline - 5:
        pts, energy = _local_refine(best_pts, max_iters=300, tol=1e-13)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final check
    if time.time() - start_time < deadline:
        final_pts = _project_to_sphere(best_pts)
        # Validate
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(final_pts[i] - final_pts[j]) < 1e-9:
                    # Points too close, try to fix
                    final_pts[i] = final_pts[i] * 1.0 + np.random.randn(3) * 1e-6
                    final_pts = _project_to_sphere(final_pts)
        
        # Verify finite
        if not np.all(np.isfinite(final_pts)):
            return _fibonacci_sphere(N)
        
        return final_pts
    
    return best_pts

# EVOLVE-BLOCK-END
