"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            total += math.log(1.0 / d)
    return total


def _log_energy_grad(points):
    """Compute gradient of logarithmic energy w.r.t. points."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            term = -diff / (d * d)
            grad[i] += term
            grad[j] -= term
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


def _initial_random():
    """Generate random points on the sphere."""
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _optimize_from(points, method='L-BFGS-B', maxiter=5000):
    """Optimize points using gradient-based methods with sphere projection."""
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _log_energy_grad(pts)
        for i in range(N):
            g[i] -= np.dot(g[i], pts[i]) * pts[i]
        return g.flatten()
    
    x0 = points.flatten()
    res = minimize(objective, x0, jac=gradient, method=method,
                   options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10})
    pts = _project_to_sphere(res.x.reshape(N, 3))
    return pts, res.fun


def _local_refinement(points, n_iter=2000, step_scale=0.1):
    """Perform local refinement using projected gradient descent."""
    pts = points.copy()
    for it in range(n_iter):
        grad = _log_energy_grad(pts)
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        step = step_scale / (1.0 + 0.01 * it)
        new_pts = pts - step * grad
        new_pts = _project_to_sphere(new_pts)
        old_e = _log_energy(pts)
        new_e = _log_energy(new_pts)
        if new_e < old_e:
            pts = new_pts
        else:
            step_scale *= 0.9
        if step_scale < 1e-8:
            break
    return pts


def _simulated_annealing(points, n_iter=30000):
    """Simulated annealing with incremental energy updates."""
    pts = points.copy()
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T_start = 0.5
    T_end = 1e-6
    rng = np.random.default_rng(123)
    
    for it in range(n_iter):
        progress = it / n_iter
        T = T_start * (T_end / T_start) ** progress
        
        idx = rng.integers(0, N)
        p = pts[idx]
        
        v = rng.randn(3)
        dot_vp = np.dot(v, p)
        t = v - dot_vp * p
        norm_t = np.linalg.norm(t)
        if norm_t < 1e-10:
            continue
        axis = t / norm_t
        
        angle = 0.3 * (T / T_start) + 0.005
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Rodrigues rotation
        new_p = p * cos_a + np.cross(axis, p) * sin_a + axis * np.dot(axis, p) * (1 - cos_a)
        new_p = new_p / np.linalg.norm(new_p)
        
        # Incremental energy change
        delta_e = 0.0
        for j in range(N):
            if j == idx:
                continue
            old_d = np.linalg.norm(p - pts[j])
            new_d = np.linalg.norm(new_p - pts[j])
            if old_d < 1e-15 or new_d < 1e-15:
                delta_e = 1e10
                break
            delta_e += math.log(1.0 / new_d) - math.log(1.0 / old_d)
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            pts[idx] = new_p
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
    
    return best_pts, best_energy


def construct():
    # Generate multiple initial configurations
    init_fib = _initial_fibonacci()
    init_rand = _initial_random()
    
    # Try multiple optimization strategies
    best_pts = None
    best_energy = float('inf')
    
    candidates = [init_fib, init_rand]
    
    for init in candidates:
        # L-BFGS-B optimization
        try:
            pts, energy = _optimize_from(init, method='L-BFGS-B', maxiter=3000)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            pass
        
        # Local refinement
        try:
            pts_ref = _local_refinement(pts, n_iter=1000)
            e_ref = _log_energy(pts_ref)
            if e_ref < best_energy:
                best_energy = e_ref
                best_pts = pts_ref.copy()
        except Exception:
            pass
    
    # Simulated annealing from best
    if best_pts is not None:
        try:
            sa_pts, sa_energy = _simulated_annealing(best_pts, n_iter=30000)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
        except Exception:
            pass
    
    # Final refinement
    if best_pts is not None:
        try:
            final_pts = _local_refinement(best_pts, n_iter=500, step_scale=0.05)
            e_final = _log_energy(final_pts)
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
                best_pts[j] += 1e-6 * np.random.randn(3)
                best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
