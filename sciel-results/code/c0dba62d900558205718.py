"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_DEADLINE = 115.0  # seconds; safety margin before 120s kill

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
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Only upper triangle
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
    """Compute gradient of energy w.r.t. point positions (before projection)."""
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
            # d/dx_i of log(1/|x_i - x_j|) = -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j|^2
            # = -(x_i - x_j) / d^2
            grad[i] += -diff / (d * d)
    return grad

def _cg_optimize(pts, max_iter=2000, tol=1e-10):
    """Polak-Ribiere conjugate gradient optimization on the sphere."""
    pts = _project(pts.copy())
    E = _energy_fast(pts)
    g = _gradient(pts)
    # Project gradient to tangent space
    g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
    s = -g_tan
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        # Line search
        alpha = 0.01
        for _ in range(30):
            new_pts = _project(pts + alpha * s)
            new_E = _energy_fast(new_pts)
            if new_E < E:
                break
            alpha *= 0.5
        else:
            break
        
        pts = new_pts
        E = new_E
        
        new_g = _gradient(pts)
        new_g_tan = new_g - np.sum(new_g * pts, axis=1, keepdims=True) * pts
        
        if np.linalg.norm(new_g_tan) < tol:
            break
        
        beta = np.dot(new_g_tan, new_g_tan - g_tan) / (np.dot(g_tan, g_tan) + 1e-15)
        beta = max(0, beta)
        s = -new_g_tan + beta * s
        g_tan = new_g_tan
    
    return pts, E

def _random_start(seed):
    """Generate a random set of points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project(pts)

def _fibonacci_start():
    """Golden-angle Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _project(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _jittered_fibonacci(jitter, seed):
    """Fibonacci start with small random jitter."""
    pts = _fibonacci_start()
    rng = np.random.default_rng(seed)
    pts = pts + rng.standard_normal((N, 3)) * jitter
    return _project(pts)

def construct():
    start_time = time.time()
    
    best_pts = None
    best_E = 1e18
    
    # Try multiple diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(('fib', _fibonacci_start()))
    
    # 2. Random starts
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > _DEADLINE - 5:
            break
        initializations.append((f'rand_{seed}', _random_start(seed)))
    
    # 3. Jittered Fibonacci
    for jitter in [0.01, 0.05, 0.1]:
        if time.time() > _DEADLINE - 5:
            break
        initializations.append((f'jitter_{jitter}', _jittered_fibonacci(jitter, seed=77)))
    
    for name, init_pts in initializations:
        if time.time() > _DEADLINE - 3:
            break
        
        opt_pts, E = _cg_optimize(init_pts, max_iter=1500)
        
        if E < best_E:
            best_E = E
            best_pts = opt_pts
    
    if best_pts is None:
        # Fallback: return Fibonacci if something went wrong
        return _fibonacci_start()
    
    return _project(best_pts)
# EVOLVE-BLOCK-END
