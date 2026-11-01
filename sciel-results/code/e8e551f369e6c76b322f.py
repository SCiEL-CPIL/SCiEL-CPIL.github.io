"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time


def _log_energy_vectorized(points):
    """Compute logarithmic energy using vectorized NumPy operations."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Use upper triangle only (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)


def _log_energy_grad_vectorized(points):
    """Compute gradient of logarithmic energy using vectorized operations."""
    n = len(points)
    grad = np.zeros_like(points)
    # Compute all pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Avoid division by zero
    dist = np.maximum(dist, 1e-15)
    # term[i,j] = -(points[i] - points[j]) / dist[i,j]^2
    term = -diff / (dist ** 2)[:, :, np.newaxis]
    # For each i, sum over j != i
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        grad[i] = np.sum(term[i, mask], axis=0)
    return grad


def _project_to_sphere(points):
    """Project each point to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms


def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _initial_random(seed=42):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _optimize_l_bfgs(points, maxiter=5000):
    """Optimize points using L-BFGS-B with vectorized gradient."""
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy_vectorized(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _log_energy_grad_vectorized(pts)
        # Project gradient to tangent space
        for i in range(N):
            g[i] -= np.dot(g[i], pts[i]) * pts[i]
        return g.flatten()
    
    x0 = points.flatten()
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10})
    pts = _project_to_sphere(res.x.reshape(N, 3))
    return pts, res.fun


def _gradient_descent_refine(points, n_iter=3000, initial_step=0.05):
    """Perform local refinement using projected gradient descent with backtracking."""
    pts = points.copy()
    step = initial_step
    for it in range(n_iter):
        grad = _log_energy_grad_vectorized(pts)
        # Project to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        new_pts = pts - step * grad
        new_pts = _project_to_sphere(new_pts)
        
        old_e = _log_energy_vectorized(pts)
        new_e = _log_energy_vectorized(new_pts)
        
        if new_e < old_e:
            pts = new_pts
            step = min(step * 1.05, 0.1)
        else:
            step *= 0.7
            if step < 1e-10:
                break
    return pts


def _simulated_annealing_refine(points, n_iter=2000, T0=0.1, T_end=0.001):
    """Refine using simulated annealing to escape local minima."""
    pts = points.copy()
    current_e = _log_energy_vectorized(pts)
    best_pts = pts.copy()
    best_e = current_e
    
    for it in range(n_iter):
        # Cooling schedule
        T = T0 * (T_end / T0) ** (it / max(n_iter - 1, 1))
        
        # Generate random perturbation
        rng = np.random.default_rng(it)
        perturb = rng.normal(0, T, size=(N, 3))
        new_pts = pts + perturb
        new_pts = _project_to_sphere(new_pts)
        
        new_e = _log_energy_vectorized(new_pts)
        delta_e = new_e - current_e
        
        # Accept if better or with probability exp(-delta_e/T)
        if delta_e < 0 or (T > 0 and rng.random() < math.exp(-delta_e / T)):
            pts = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_pts = pts.copy()
    
    return best_pts


def construct():
    start_time = time.time()
    
    # Generate multiple initial configurations
    init_fib = _initial_fibonacci()
    init_rand1 = _initial_random(seed=42)
    init_rand2 = _initial_random(seed=123)
    
    best_pts = None
    best_energy = float('inf')
    
    candidates = [init_fib, init_rand1, init_rand2]
    
    for init in candidates:
        # L-BFGS-B optimization
        try:
            pts, energy = _optimize_l_bfgs(init, maxiter=3000)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            pass
        
        # Gradient descent refinement
        try:
            pts_ref = _gradient_descent_refine(pts, n_iter=2000)
            e_ref = _log_energy_vectorized(pts_ref)
            if e_ref < best_energy:
                best_energy = e_ref
                best_pts = pts_ref.copy()
        except Exception:
            pass
        
        # Simulated annealing for escaping local minima
        try:
            pts_sa = _simulated_annealing_refine(pts, n_iter=1500)
            e_sa = _log_energy_vectorized(pts_sa)
            if e_sa < best_energy:
                best_energy = e_sa
                best_pts = pts_sa.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() - start_time > 90:
            break
    
    # Final refinement with gradient descent
    if best_pts is not None:
        try:
            final_pts = _gradient_descent_refine(best_pts, n_iter=2000, initial_step=0.02)
            e_final = _log_energy_vectorized(final_pts)
            if e_final < best_energy:
                best_pts = final_pts
                best_energy = e_final
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb
                rng = np.random.default_rng(42 + i)
                best_pts[j] += 1e-6 * rng.randn(3)
                best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
