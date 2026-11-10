"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(pts):
    """Compute gradient of log energy. d/dx_i sum_{j != i} log(1/|x_i - x_j|)
    = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2"""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-18:
                continue
            d = math.sqrt(d2)
            grad[i] += diff / d2
    return grad


def _project_to_sphere(pts):
    """Project each point to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _log_energy_smooth(pts):
    """Smoothed log energy for optimization: log(1/d) with a small epsilon floor on d."""
    n = len(pts)
    total = 0.0
    eps = 1e-12
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            d = max(d, eps)
            total += math.log(1.0 / d)
    return total


def _log_energy_grad_smooth(pts):
    """Gradient of smoothed log energy."""
    n = len(pts)
    grad = np.zeros_like(pts)
    eps = 1e-12
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            d = max(d, eps)
            grad[i] += diff / (d * d)
    return grad


def _fibonacci_sphere(n):
    """Generate n points on unit sphere via golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(pts, deadline, label=""):
    """Run L-BFGS-B optimization starting from pts, respecting deadline."""
    t0 = time.time()
    
    # Flatten for scipy
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        return _log_energy_smooth(p)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        g = _log_energy_grad_smooth(p)
        # Project gradient to tangent space
        g = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g.flatten()
    
    try:
        res = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 50000}
        )
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 5:
        return pts
    
    best_pts = _project_to_sphere(res.x.reshape(-1, 3))
    return best_pts


def _high_precision_polish(pts, deadline):
    """Final high-precision polish with tighter tolerances."""
    t0 = time.time()
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        return _log_energy_smooth(p)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        g = _log_energy_grad_smooth(p)
        g = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g.flatten()
    
    try:
        res = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 100000}
        )
        best_pts = _project_to_sphere(res.x.reshape(-1, 3))
    except Exception:
        best_pts = pts
    
    return best_pts


def construct():
    deadline = time.time() + 140  # 140s safety margin within 150s limit
    
    # Generate multiple diverse starting points
    starts = []
    
    # 1. Fibonacci spiral (deterministic, good spread)
    starts.append(_fibonacci_sphere(N))
    
    # 2. Random starts with different seeds
    for seed in [42, 123, 777, 2024]:
        starts.append(_random_sphere(N, seed=seed))
    
    # 3. Slightly perturbed Fibonacci
    rng = np.random.default_rng(999)
    fib = _fibonacci_sphere(N)
    perturbed = fib + rng.normal(0, 0.1, size=fib.shape)
    starts.append(_project_to_sphere(perturbed))
    
    # 4. Another perturbation
    perturbed2 = fib + rng.normal(0, 0.05, size=fib.shape)
    starts.append(_project_to_sphere(perturbed2))
    
    # Optimize each start, track best
    best_pts = None
    best_energy = float('inf')
    
    for idx, start in enumerate(starts):
        if time.time() > deadline - 20:
            break
        
        # Quick check initial energy
        e_init = _log_energy_smooth(start)
        
        # Optimize
        result = _optimize_from_sphere(start, deadline)
        
        # Evaluate with precise energy
        e_result = _log_energy(result)
        
        if e_result < best_energy and e_result < float('inf'):
            best_energy = e_result
            best_pts = result
    
    # If we have time, do high-precision polish on the best
    if best_pts is not None and time.time() < deadline - 10:
        polished = _high_precision_polish(best_pts, deadline)
        e_polished = _log_energy(polished)
        if e_polished < best_energy:
            best_pts = polished
            best_energy = e_polished
    
    # Final validation: ensure no points too close
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Verify no points closer than 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them
                mid = (best_pts[i] + best_pts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                best_pts[i] = best_pts[i] + 1e-8 * mid
                best_pts[j] = best_pts[j] - 1e-8 * mid
                best_pts[i] = best_pts[i] / np.linalg.norm(best_pts[i])
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts


def _optimize_from_sphere(pts, deadline):
    """Run L-BFGS-B optimization starting from pts, respecting deadline."""
    t0 = time.time()
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        return _log_energy_smooth(p)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        g = _log_energy_grad_smooth(p)
        g = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g.flatten()
    
    try:
        res = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 50000}
        )
        best_pts = _project_to_sphere(res.x.reshape(-1, 3))
    except Exception:
        best_pts = pts
    
    return best_pts

# EVOLVE-BLOCK-END
