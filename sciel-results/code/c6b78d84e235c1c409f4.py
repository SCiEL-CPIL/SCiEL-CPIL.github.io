"""
Combined program: Hybrid L-BFGS-B and Riemannian Conjugate Gradient optimization
for minimal logarithmic energy of 150 points on the unit sphere.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0  # seconds before the deadline to stop and return best


def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient with respect to the points, projected to the tangent space.
    
    Vectorized implementation for speed.
    """
    n = pts.shape[0]
    
    # Compute pairwise differences and squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient calculation:
    # grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # = sum_{j != i} x_j * inv_d2[i,j] - x_i * sum_{j != i} inv_d2[i,j]
    # = (inv_d2 @ pts)_i - x_i * (sum(inv_d2, axis=1))_i
    
    weighted_sum_pts = inv_d2 @ pts  # (n, 3)
    sum_inv_d2 = np.sum(inv_d2, axis=1)  # (n,)
    grad = weighted_sum_pts - pts * sum_inv_d2[:, np.newaxis]
    
    # Project gradient to tangent space: g_tangent = g - (g . x) * x
    dot_prod = np.sum(grad * pts, axis=1)  # (n,)
    grad = grad - dot_prod[:, np.newaxis] * pts
    
    # Energy calculation: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)  # Prevent log(0)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed (no gradient)."""
    n = pts.shape[0]
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        u3 = halton(i, base_start + 2)
        
        # Map to sphere using spherical coordinates
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    # For unit sphere, exponential map: exp_p(v) = cos(|v|)p + sin(|v|)/|v| v
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs


def _riemannian_cg(pts, time_limit=15.0, max_iter=500):
    """
    Riemannian conjugate gradient on the sphere using exponential maps.
    This is a first-order method that respects the manifold geometry.
    """
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # Initial search direction = negative gradient (already in tangent space)
    d = -grad.copy()
    prev_grad = None
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            # Armijo condition check (simplified)
            if new_energy > energy:
                break
            
            alpha *= 0.8
        
        if best_alpha == 0.0:
            break
        
        # Update
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Polak-Ribiere beta for conjugate gradient
        d_new = -grad
        beta = 0.0
        if prev_grad is not None:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
        
        d = d_new + beta * d
        prev_grad = grad.copy()
    
    return current, energy


def _lbfgs_optimize(pts, time_limit=30.0):
    """
    Optimize using L-BFGS-B with projected gradient.
    This is a second-order method that is robust for initial convergence.
    """
    start = time.time()
    
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        return energy, grad.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxfun': 10000, 'maxls': 50})
        
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        energy = _log_energy_fast(x_opt)
        
        return x_opt, energy
    except Exception:
        return pts, _log_energy_fast(pts)


def _dual_optimize(pts, time_limit=40.0):
    """
    Alternate between L-BFGS-B and Riemannian CG.
    Phase 1: L-BFGS-B for coarse optimization (fast initial convergence)
    Phase 2: Riemannian CG for fine optimization (manifold-aware polish)
    Phase 3: Final L-BFGS-B polish (optional, if time permits)
    """
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Phase 1: L-BFGS-B for coarse optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 10:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left * 0.6, 25.0))
    
    # Phase 2: Riemannian CG for fine optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _riemannian_cg(current, time_limit=min(time_left * 0.8, 15.0))
    
    # Phase 3: Final L-BFGS-B polish
    time_left = time_limit - (time.time() - start)
    if time_left > 3:
        current, energy = _lbfgs_optimize(current, time_limit=min(time_left, 10.0))
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    best_pts = None
    best_energy = float('inf')
    
    # Try multiple initializations
    initializations = []
    
    # Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib)
    
    # Halton sequence
    pts_halton = _halton_sphere(N)
    initializations.append(pts_halton)
    
    # Random initialization
    np.random.seed(42)
    pts_rand = np.random.randn(N, 3)
    pts_rand = _normalize(pts_rand)
    initializations.append(pts_rand)
    
    # Calculate time budget per initialization
    time_per_init = (DEADLINE_SECONDS - SAFETY_MARGIN) / len(initializations)
    
    for init_pts in initializations:
        if time.time() > deadline:
            break
        
        # Evaluate initial energy
        e0, _ = _energy_and_grad(init_pts)
        
        if e0 < best_energy:
            best_energy = e0
            best_pts = init_pts.copy()
        
        # Run dual optimization (L-BFGS-B + RCG)
        try:
            pts_opt, e_opt = _dual_optimize(init_pts, time_limit=time_per_init - 1.0)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
    
    # Final projection and validation
    best_pts = _normalize(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                best_pts[j] += 1e-8 * np.random.randn(3)
                best_pts[j] = _normalize(best_pts[j])
    
    return best_pts
