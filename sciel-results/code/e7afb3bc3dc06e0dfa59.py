"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START

def _energy_fast(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) using vectorized ops."""
    # ||xi - xj||^2 = 2 - 2 xi.xj
    sq_dists = 2.0 - 2.0 * (pts @ pts.T)
    np.fill_diagonal(sq_dists, 1.0)
    # Clip small negative values from numerical error
    sq_dists = np.maximum(sq_dists, 1e-15)
    # Upper triangle indices
    iu = np.triu_indices(N, k=1)
    # log(1/d) = -0.5 * log(d^2)
    log_dists_sq = np.log(sq_dists[iu])
    return -0.5 * math.fsum(log_dists_sq.tolist())

def _gradient(pts):
    """Compute gradient of logarithmic energy, projected onto tangent space."""
    # diffs[i, j] = pts[i] - pts[j]
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dists_sq[i, j] = ||pts[i] - pts[j]||^2
    dists_sq = np.sum(diffs**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    # weights[i, j] = 1 / dists_sq[i, j]
    weights = 1.0 / dists_sq
    np.fill_diagonal(weights, 0.0)
    # grad_raw[i] = sum_j (pts[i] - pts[j]) * weights[i, j]
    grad_raw = np.sum(diffs * weights[..., np.newaxis], axis=1)
    
    # Project onto tangent space: g - (g . x) x
    dot_products = np.sum(grad_raw * pts, axis=1)
    grad = grad_raw - dot_products[:, np.newaxis] * pts
    return grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _initial_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _adam_optimize(pts, max_iter=10000, time_limit=None, start_time=None, lr_init=0.05):
    """Adam optimizer for sphere-constrained optimization."""
    if start_time is None:
        start_time = time.time()
    
    pts = _project_to_sphere(pts.copy())
    lr = lr_init
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    best_energy = _energy_fast(pts)
    best_pts = pts.copy()
    
    # Adaptive learning rate decay
    decay = 0.999
    
    for i in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        grad = _gradient(pts)
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update
        pts_new = pts - lr * m_hat / (np.sqrt(v_hat) + eps)
        pts_new = _project_to_sphere(pts_new)
        
        energy = _energy_fast(pts_new)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_new.copy()
        
        pts = pts_new
        lr *= decay
        
    return best_pts, best_energy

def construct():
    deadline = time.time() + 145  # Leave 5s margin for 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Fibonacci start
    pts = _initial_fibonacci(N)
    t0 = time.time()
    opts = _adam_optimize(pts, max_iter=5000, time_limit=60, start_time=t0, lr_init=0.05)
    if opts[1] < best_energy:
        best_energy = opts[1]
        best_pts = opts[0]
        
    # Phase 2: Random restarts
    # Use remaining time for random restarts
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        num_restarts = max(1, int(remaining_time / 20)) # ~20s per restart
        for k in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = 42 + k * 1000
            pts_r = _initial_random(N, seed=seed)
            t0 = time.time()
            # Limit time per restart to ensure we finish
            time_left = deadline - time.time() - 2
            opts = _adam_optimize(pts_r, max_iter=3000, time_limit=time_left, start_time=t0, lr_init=0.05)
            if opts[1] < best_energy:
                best_energy = opts[1]
                best_pts = opts[0]
                
    # Phase 3: Polish best solution if time remains
    if best_pts is not None and time.time() < deadline - 5:
        t0 = time.time()
        time_left = deadline - time.time() - 2
        # Use smaller LR for polishing
        opts = _adam_optimize(best_pts, max_iter=2000, time_limit=time_left, start_time=t0, lr_init=0.01)
        if opts[1] < best_energy:
            best_energy = opts[1]
            best_pts = opts[0]
            
    return _project_to_sphere(best_pts)
# EVOLVE-BLOCK-END
