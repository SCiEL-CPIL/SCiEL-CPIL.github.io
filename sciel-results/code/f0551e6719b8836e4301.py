"""Improved program for logarithmic energy minimization on the sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, Bounds

_DEADLINE = 145.0  # seconds; safety margin before 150s kill

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return 1e18
            total += math.log(1.0 / d)
    return total

def _energy_fast(pts):
    """Vectorized energy computation."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    if np.any(d < 1e-9):
        return 1e18
    return float(np.sum(np.log(1.0 / d)))

def _project(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _gradient(pts):
    """Compute gradient of energy w.r.t. point positions."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-9:
                continue
            grad[i] += -diff / (d * d)
    return grad

def _tangent_gradient(pts):
    """Compute gradient projected onto tangent space."""
    g = _gradient(pts)
    g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
    return g_tan

def _fibonacci_start():
    """Golden-angle Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _project(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _random_start(seed):
    """Generate a random set of points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project(pts)

def _jittered_fibonacci(jitter, seed):
    """Fibonacci start with small random jitter."""
    pts = _fibonacci_start()
    rng = np.random.default_rng(seed)
    pts = pts + rng.standard_normal((N, 3)) * jitter
    return _project(pts)

def _l_bfgs_optimize(pts, max_iter=5000, tol=1e-12):
    """L-BFGS-B optimization on the sphere using tangent space parameterization."""
    pts = _project(pts.copy())
    
    def _objective_and_grad(flat):
        """Objective and gradient in flattened tangent space coordinates."""
        # Reconstruct points from tangent space offsets
        # We'll use a simple approach: treat each point independently
        # For L-BFGS-B, we need to work with a flat parameter vector
        # Strategy: use spherical coordinates or just project after each step
        
        # Actually, let's use a different approach:
        # Work in Cartesian coordinates but project gradient to tangent space
        # and use L-BFGS-B with bounds
        pass
    
    # Simpler approach: use scipy's L-BFGS-B directly on Cartesian coordinates
    # with a penalty for leaving the sphere, or just project at each iteration
    
    # Let's use a cleaner approach: optimize in Cartesian with projection
    best_pts = pts.copy()
    best_E = _energy_fast(pts)
    
    # Use L-BFGS-B with a wrapper that projects to sphere
    def _wrapped_objective(x):
        """Objective function for L-BFGS-B."""
        pts_x = _project(x.reshape(N, 3))
        return _energy_fast(pts_x)
    
    def _wrapped_gradient(x):
        """Gradient for L-BFGS-B (projected to tangent space)."""
        pts_x = _project(x.reshape(N, 3))
        g = _tangent_gradient(pts_x)
        return g.flatten()
    
    # Initial guess
    x0 = pts.flatten()
    
    # Run L-BFGS-B
    try:
        result = minimize(
            _wrapped_objective,
            x0,
            jac=_wrapped_gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
        )
        
        if result.success or result.fun < best_E:
            best_pts = _project(result.x.reshape(N, 3))
            best_E = _energy_fast(best_pts)
    except Exception:
        pass
    
    return best_pts, best_E

def _conjugate_gradient_optimize(pts, max_iter=3000, tol=1e-12):
    """Polak-Ribiere conjugate gradient optimization on the sphere."""
    pts = _project(pts.copy())
    E = _energy_fast(pts)
    g = _tangent_gradient(pts)
    s = -g
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 2:
            break
        
        # Line search (backtracking)
        alpha = 0.01
        improved = False
        for _ in range(40):
            new_pts = _project(pts + alpha * s)
            new_E = _energy_fast(new_pts)
            if new_E < E:
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
        
        pts = new_pts
        E = new_E
        
        new_g = _tangent_gradient(pts)
        
        if np.linalg.norm(new_g) < tol:
            break
        
        # Polak-Ribiere beta
        beta_num = np.dot(new_g, new_g - g)
        beta_den = np.dot(g, g)
        beta = max(0, beta_num / (beta_den + 1e-15))
        s = -new_g + beta * s
        g = new_g
    
    return pts, E

def _multi_start_optimize():
    """Multi-start optimization with diverse initializations."""
    start_time = time.time()
    
    best_pts = None
    best_E = 1e18
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral (usually good starting point)
    initializations.append(_fibonacci_start())
    
    # 2. Jittered Fibonacci with different jitter levels
    for jitter in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        if time.time() > _DEADLINE - 10:
            break
        initializations.append(_jittered_fibonacci(jitter, seed=hash((jitter,)) % 2**32))
    
    # 3. Random starts with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        if time.time() > _DEADLINE - 10:
            break
        initializations.append(_random_start(seed))
    
    # 4. Structured starts: latitude/longitude grids
    for lat_div, lon_div in [(4, 25), (5, 20), (8, 12)]:
        if time.time() > _DEADLINE - 10:
            break
        pts = []
        for lat_idx in range(lat_div):
            lat = -np.pi/2 + np.pi * (lat_idx + 0.5) / lat_div
            cos_lat = np.cos(lat)
            sin_lat = np.sin(lat)
            for lon_idx in range(lon_div):
                lon = 2 * np.pi * (lon_idx + 0.5) / lon_div
                x = cos_lat * np.cos(lon)
                y = cos_lat * np.sin(lon)
                z = sin_lat
                pts.append([x, y, z])
        if len(pts) == N:
            initializations.append(_project(np.array(pts)))
    
    # Optimize from each initialization
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > _DEADLINE - 5:
            break
        
        # First try conjugate gradient (fast)
        if time.time() < _DEADLINE - 30:
            opt_pts, E = _conjugate_gradient_optimize(init_pts, max_iter=2000)
            if E < best_E:
                best_E = E
                best_pts = opt_pts.copy()
        
        # Then try L-BFGS-B from the CG result (more precise)
        if best_pts is not None and time.time() < _DEADLINE - 15:
            opt_pts, E = _l_bfgs_optimize(best_pts, max_iter=3000)
            if E < best_E:
                best_E = E
                best_pts = opt_pts.copy()
    
    # Final polish: run L-BFGS-B with very tight tolerance from best result
    if best_pts is not None and time.time() < _DEADLINE - 5:
        final_pts, final_E = _l_bfgs_optimize(best_pts, max_iter=10000, tol=1e-14)
        if final_E < best_E:
            best_E = final_E
            best_pts = final_pts
    
    if best_pts is None:
        return _fibonacci_start()
    
    return _project(best_pts)

def construct():
    """Main entry point: minimize logarithmic energy of 100 points on sphere."""
    start_time = time.time()
    
    try:
        result = _multi_start_optimize()
    except Exception:
        # Fallback to Fibonacci if something goes wrong
        result = _fibonacci_start()
    
    # Ensure valid output
    result = _project(result)
    
    # Verify no points are too close
    n = len(result)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                # Add small perturbation
                rng = np.random.default_rng(12345)
                result[i] = _project(result[i] + rng.standard_normal(3) * 1e-6)
    
    return result
# EVOLVE-BLOCK-END
