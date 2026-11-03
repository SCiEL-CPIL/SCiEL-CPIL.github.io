"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _remaining(deadline, margin=10.0):
    """Return remaining time in seconds before deadline minus margin."""
    return max(0.0, deadline - time.time() - margin)

def _normalize(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms

def _energy_fast(pts):
    """Compute logarithmic energy efficiently."""
    pts = np.asarray(pts)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _energy_grad(pts):
    """Compute energy and gradient."""
    pts = np.asarray(pts)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = diff / dist_sq[:, :, np.newaxis]
    idx = np.diag_indices(n)
    terms[idx[0], idx[1], :] = 0.0
    grad = np.sum(terms, axis=1)
    return energy, grad

def _precise_energy(pts):
    """Compute energy using math.fsum for precision."""
    import math
    pts = np.asarray(pts)
    n = len(pts)
    total = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return -np.inf
            total.append(math.log(1.0 / d))
    return math.fsum(total)

def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _stratified_sphere(n, seed=None):
    """Generate stratified points on sphere for better coverage."""
    if seed is not None:
        np.random.seed(seed)
    # Use spherical coordinates with stratified sampling
    z = 1.0 - 2.0 * (np.arange(n) + 0.5) / n
    theta = 2.0 * np.pi * (np.arange(n) + 0.5) / n + np.random.uniform(0, 2*np.pi, n)
    r = np.sqrt(np.maximum(1.0 - z*z, 0.0))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

def _random_sphere(n, seed=None):
    """Generate random points on the sphere."""
    if seed is not None:
        np.random.seed(seed)
    v = np.random.randn(n, 3)
    return _normalize(v)

def _exp_map(pts, vec, step=1.0):
    """
    Exponential map on the sphere: move pts along vec (tangent vectors) by step.
    For each point p and tangent vector v, exp_p(step*v) is computed via
    Rodrigues' rotation formula.
    """
    pts = np.asarray(pts)
    vec = np.asarray(vec)
    n = len(pts)
    # Project vec to tangent space
    vec_tan = vec - np.sum(vec * pts, axis=1, keepdims=True) * pts
    # Compute angle
    theta = step * np.linalg.norm(vec_tan, axis=1)
    theta = np.maximum(theta, 1e-15)
    # Unit vector for rotation axis
    axis = vec_tan / theta[:, np.newaxis]
    # Rodrigues' formula
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # p' = p*cos(theta) + axis*sin(theta) + (p x axis)*(1-cos(theta))
    cross = np.cross(pts, axis)
    new_pts = pts * cos_t[:, np.newaxis] + axis * sin_t[:, np.newaxis] + cross * (1 - cos_t)[:, np.newaxis]
    return _normalize(new_pts)

def _riemannian_polish(pts, max_iter=300, tol=1e-14, deadline=None):
    """
    Riemannian gradient descent with backtracking line search for final polish.
    """
    pts = _normalize(pts.copy())
    best_energy = _energy_fast(pts)
    best_pts = pts.copy()
    lr = 0.1
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_grad(pts)
        # Project gradient to tangent space
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        grad_norm = np.linalg.norm(g_tan)
        
        if grad_norm < tol:
            break
        
        # Backtracking line search
        step = lr
        improved = False
        for backtrack in range(20):
            new_pts = _exp_map(pts, g_tan, step=step)
            new_e = _energy_fast(new_pts)
            
            if new_e < e - 1e-10 * step * grad_norm:
                pts = new_pts
                best_energy = new_e
                best_pts = new_pts.copy()
                improved = True
                break
            else:
                step *= 0.5
        
        if not improved:
            # Try smaller step
            step = 1e-6
            new_pts = _exp_map(pts, g_tan, step=step)
            new_e = _energy_fast(new_pts)
            if new_e < e:
                pts = new_pts
                best_energy = new_e
                best_pts = new_pts.copy()
            else:
                break
        
        # Adaptive learning rate
        if improved:
            lr = min(lr * 1.1, 1.0)
        else:
            lr *= 0.5
    
    return best_pts, best_energy

def _newton_polish(pts, max_iter=100, tol=1e-14, deadline=None):
    """
    Newton-type method using L-BFGS with very tight tolerances.
    """
    pts = _normalize(pts.copy())
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 3)
        p = _normalize(p)
        return _energy_fast(p)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        p = _normalize(p)
        e, g = _energy_grad(p)
        g_tan = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tan.flatten()
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    
    pts_opt = _normalize(result.x.reshape(-1, 3))
    return pts_opt, result.fun

def _optimize_from_start(pts0, deadline, margin=10.0):
    """
    Full optimization pipeline from a starting configuration.
    """
    pts = _normalize(pts0.copy())
    current_energy = _energy_fast(pts)
    
    # Phase 1: L-BFGS-B with moderate tolerance
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(-1, 3)
        p = _normalize(p)
        return _energy_fast(p)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        p = _normalize(p)
        e, g = _energy_grad(p)
        g_tan = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tan.flatten()
    
    remaining = _remaining(deadline, margin)
    if remaining > 5:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 50}
        )
        pts = _normalize(result.x.reshape(-1, 3))
        current_energy = _energy_fast(pts)
    
    # Phase 2: Riemannian polish with backtracking
    remaining = _remaining(deadline, margin)
    if remaining > 2:
        pts, current_energy = _riemannian_polish(pts, max_iter=200, tol=1e-13, deadline=deadline - margin)
    
    # Phase 3: Final L-BFGS polish with tight tolerance
    remaining = _remaining(deadline, margin)
    if remaining > 1:
        pts, current_energy = _newton_polish(pts, max_iter=50, tol=1e-14, deadline=deadline - margin)
    
    # Phase 4: Final Riemannian micro-polish
    remaining = _remaining(deadline, margin)
    if remaining > 0.5:
        pts, current_energy = _riemannian_polish(pts, max_iter=50, tol=1e-15, deadline=deadline - margin)
    
    return pts, current_energy

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s limit
    
    np.random.seed(42)
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_fibonacci_spiral(N))
    inits.append(_stratified_sphere(N, seed=42))
    inits.append(_random_sphere(N, seed=123))
    inits.append(_random_sphere(N, seed=456))
    inits.append(_random_sphere(N, seed=789))
    
    # Add perturbed versions of Fibonacci
    for scale in [0.03, 0.07, 0.15, 0.3]:
        perturbed = _fibonacci_spiral(N) + scale * np.random.randn(N, 3)
        inits.append(_normalize(perturbed))
    
    # Add perturbed stratified
    for scale in [0.05, 0.1]:
        perturbed = _stratified_sphere(N, seed=42) + scale * np.random.randn(N, 3)
        inits.append(_normalize(perturbed))
    
    best_pts = None
    best_energy = np.inf
    
    for i, init in enumerate(inits):
        remaining = _remaining(deadline, margin=10.0)
        if remaining < 8.0:
            break
        
        pts_opt, energy = _optimize_from_start(init, deadline, margin=10.0)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
    
    # If we haven't found a good solution, use the first one
    if best_pts is None:
        best_pts, best_energy = _optimize_from_start(inits[0], deadline, margin=10.0)
    
    # Final intensive polish on the best solution
    remaining = _remaining(deadline, margin=5.0)
    if remaining > 2.0:
        best_pts, best_energy = _riemannian_polish(best_pts, max_iter=100, tol=1e-14, deadline=deadline - 5.0)
    
    remaining = _remaining(deadline, margin=5.0)
    if remaining > 1.0:
        best_pts, best_energy = _newton_polish(best_pts, max_iter=30, tol=1e-14, deadline=deadline - 5.0)
    
    remaining = _remaining(deadline, margin=3.0)
    if remaining > 0.5:
        best_pts, best_energy = _riemannian_polish(best_pts, max_iter=30, tol=1e-15, deadline=deadline - 3.0)
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
