import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 150.0
DEADLINE_FRACTION = 0.88

def _log_energy_fast(pts):
    """Vectorized log energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    iu = np.triu_indices(n, k=1)
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * np.sum(np.log(dists2[iu]))

def _log_energy_grad_fast(pts):
    """Vectorized log energy and gradient (Euclidean, not projected)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    
    # Energy: -0.5 * sum log(d^2) over upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists2[iu]))
    
    # Gradient of E = sum_{i<j} log(1/|xi-xj|) = -0.5 * sum_{i<j} log(|xi-xj|^2)
    # dE/dxi = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    inv_dists2 = 1.0 / dists2
    np.fill_diagonal(inv_dists2, 0.0)
    
    # sum_j (xi-xj)/d_ij^2
    s = (diff * inv_dists2[:, :, np.newaxis]).sum(axis=1)
    
    # Gradient of E is -s
    grad = -s
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, grad):
    """Project gradient to tangent space of sphere."""
    coeffs = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - coeffs * pts

def _fibonacci_spiral(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n=N, seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def _icosahedron_subdivide(n=N, seed=None):
    """Start from icosahedron and subdivide to get n points."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    # Use icosahedron as base, then add random points
    if n <= 12:
        return verts[:n]
    
    rng = np.random.default_rng(seed)
    extra = rng.standard_normal((n - 12, 3))
    extra = _project_to_sphere(extra)
    return np.vstack([verts, extra])

def _optimize_lbfgs(pts, deadline):
    """Run L-BFGS-B optimization with projected gradients."""
    pts = _project_to_sphere(pts)
    x0 = pts.flatten()
    
    def func(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _log_energy_grad_fast(p)
        return e
        
    def jac(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _log_energy_grad_fast(p)
        g_t = _tangent_project(p, g)
        return g_t.flatten()
        
    options = {
        'maxiter': 5000,
        'gtol': 1e-12,
        'ftol': 1e-16,
        'disp': False,
        'maxfun': 20000,
    }
    
    try:
        result = minimize(func, x0, jac=jac, method='L-BFGS-B', options=options)
        if result.fun < 1e17:
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
        
    return pts

def _polish(pts, deadline):
    """Steepest descent polish with backtracking line search."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    lr = 0.05
    min_lr = 1e-14
    max_lr = 1.0
    patience = 0
    
    while time.time() < deadline:
        e, g = _log_energy_grad_fast(pts)
        if e < -1e17: break
        
        g_t = _tangent_project(pts, g)
        
        grad_norm = np.linalg.norm(g_t)
        if grad_norm < 1e-15: break
        
        alpha = lr
        found = False
        for _ in range(50):
            new_pts = pts - alpha * g_t
            new_pts = _project_to_sphere(new_pts)
            
            new_e = _log_energy_fast(new_pts)
            
            if new_e < current_energy - 1e-18 * alpha * grad_norm:
                pts = new_pts
                current_energy = new_e
                found = True
                break
            alpha *= 0.5
        
        if found:
            patience = 0
            lr = min(alpha * 1.5, max_lr)
        else:
            lr = alpha
            patience += 1
            if patience > 80 or lr < min_lr:
                break
                
    return pts

def _newton_polish(pts, deadline):
    """High-precision Newton-style polish using local quadratic approximation."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    for step in range(500):
        if time.time() > deadline:
            break
            
        e, g = _log_energy_grad_fast(pts)
        g_t = _tangent_project(pts, g)
        grad_norm = np.linalg.norm(g_t)
        
        if grad_norm < 1e-16:
            break
        
        alpha = 0.005
        for _ in range(60):
            new_pts = pts - alpha * g_t
            new_pts = _project_to_sphere(new_pts)
            new_e = _log_energy_fast(new_pts)
            
            if new_e < current_energy - 1e-18 * alpha * grad_norm:
                pts = new_pts
                current_energy = new_e
                break
            alpha *= 0.5
        else:
            break
            
    return pts

def _simulated_annealing(pts, deadline, seed=None):
    """Simulated annealing with rotation moves to escape local minima."""
    rng = np.random.default_rng(seed)
    pts = _project_to_sphere(pts.copy())
    current_energy = _log_energy_fast(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = 0.05
    min_T = 1e-6
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Choose number of points to move
        n_moves = rng.integers(1, 4)
        indices = rng.choice(N, size=n_moves, replace=False)
        
        new_pts = pts.copy()
        for idx in indices:
            # Random rotation
            angle = rng.uniform(-np.pi, np.pi) * T
            axis = rng.standard_normal(3)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-10:
                continue
            axis = axis / axis_norm
            
            # Rodrigues rotation
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
            
            new_pts[idx] = R @ new_pts[idx]
        
        new_pts = _project_to_sphere(new_pts)
        new_energy = _log_energy_fast(new_pts)
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e/T)
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T *= 0.9998
        
    return best_pts

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    best_pts = None
    best_energy = float('inf')
    
    # Define multiple diverse starting configurations
    rng = np.random.default_rng(12345)
    starts = [
        _fibonacci_spiral(N),
        _project_to_sphere(_fibonacci_spiral(N) + rng.normal(0, 0.05, (N, 3))),
        _project_to_sphere(_fibonacci_spiral(N) + rng.normal(0, 0.15, (N, 3))),
        _project_to_sphere(_fibonacci_spiral(N) + rng.normal(0, 0.3, (N, 3))),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
        _icosahedron_subdivide(N, seed=42),
        _project_to_sphere(rng.standard_normal((N, 3))),
    ]
    
    # Remove duplicate or very similar starts
    unique_starts = [starts[0]]
    for s in starts[1:]:
        if all(np.linalg.norm(s - u) > 0.3 for u in unique_starts):
            unique_starts.append(s)
    starts = unique_starts
    
    # Allocate time: 40% optimization, 35% SA, 25% polishing
    total_time = deadline - _start_time
    opt_deadline = _start_time + total_time * 0.40
    sa_deadline = _start_time + total_time * 0.75
    polish_deadline = deadline
    
    # Phase 1: L-BFGS-B optimization from multiple starts
    for i, start in enumerate(starts):
        if time.time() > opt_deadline:
            break
            
        optimized = _optimize_lbfgs(start, opt_deadline)
        e = _log_energy_fast(optimized)
        
        if e < best_energy:
            best_energy = e
            best_pts = optimized
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _log_energy_fast(best_pts)
    
    # Phase 2: Simulated annealing to escape local minima
    if time.time() < sa_deadline:
        sa_result = _simulated_annealing(best_pts, sa_deadline, seed=999)
        sa_energy = _log_energy_fast(sa_result)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_pts = sa_result
    
    # Phase 3: Final high-precision polishing
    if time.time() < polish_deadline - 5.0:
        polished = _polish(best_pts, polish_deadline - 5.0)
        pe = _log_energy_fast(polished)
        if pe < best_energy:
            best_energy = pe
            best_pts = polished
    
    if time.time() < polish_deadline - 2.0:
        final = _newton_polish(best_pts, polish_deadline)
        fe = _log_energy_fast(final)
        if fe < best_energy:
            best_energy = fe
            best_pts = final
    
    best_pts = _project_to_sphere(best_pts)
    
    # Final validity check
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_spiral(N)
        
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    if min_d < 1e-9:
        return _fibonacci_spiral(N)
        
    return best_pts
