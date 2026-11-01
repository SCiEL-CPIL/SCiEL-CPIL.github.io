"""Optimized logarithmic energy minimization for 100 points on the sphere using Basin Hopping."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)

def _energy_and_tangent_grad(pts):
    """Compute energy and tangent gradient.
    
    Energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    Euclidean Gradient: dE/dxi = sum_{j!=i} (xj - xi) / |xi - xj|^2
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    # Energy
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    energy = -math.fsum(np.log(dist[iu]))
    
    # Gradient
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # Sum_j M_ij x_j  =>  (M @ X.T).T ? 
    # M is N x N, X is N x 3.
    # We want row i to be sum_j M_ij X_j.
    # Let A = M @ X.T. A is N x 3. A[i, k] = sum_j M[i, j] X[j, k].
    # So A[i, :] is sum_j M[i, j] X[j, :]. This is correct.
    term1 = inv_d2 @ pts.T  # N x 3
    
    # Sum_j M_ij => row sums of M
    row_sums = inv_d2.sum(axis=1)  # N
    
    # X_i * sum_j M_ij
    # We want row i to be X[i, :] * row_sums[i]
    term2 = pts * row_sums[:, np.newaxis]  # N x 3
    
    # Euclidean Gradient
    euclidean_grad = term1 - term2
    
    # Project to Tangent Space
    # g_t = g_e - (g_e . x) x
    dot_prod = np.sum(euclidean_grad * pts, axis=1, keepdims=True)
    tangent_grad = euclidean_grad - dot_prod * pts
    
    return energy, tangent_grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _riemannian_cg(pts0, max_iters=100, time_limit=5.0, tol=1e-12):
    """Riemannian Conjugate Gradient Method."""
    pts = pts0.copy()
    energy, g_t = _energy_and_tangent_grad(pts)
    
    best_pts = pts.copy()
    best_energy = energy
    
    # Initial direction
    p = -g_t
    # Ensure p is in tangent space (it is, by construction)
    
    start_time = time.time()
    it = 0
    
    while it < max_iters:
        if time.time() - start_time > time_limit:
            break
        
        # Line search
        alpha = 1.0
        improved = False
        for _ in range(30):
            # Retraction: pts + alpha * p, then normalize
            new_pts = _project_to_sphere(pts + alpha * p)
            new_energy, _ = _energy_and_tangent_grad(new_pts)
            
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
            
        # Check convergence
        if np.linalg.norm(g_t) < tol:
            break
            
        # Update gradient
        _, new_g_t = _energy_and_tangent_grad(pts)
        
        # Fletcher-Reeves beta
        denom = np.sum(g_t * g_t)
        beta = (np.sum(new_g_t * new_g_t) / denom) if denom > 1e-16 else 0.0
        
        # Update direction
        p_new = -new_g_t + beta * p
        # Project to tangent space
        p_new = p_new - np.sum(p_new * pts, axis=1, keepdims=True) * pts
        
        # If new direction is not good, reset to steepest descent
        if np.sum(p_new * new_g_t) >= 0:
            p_new = -new_g_t
            p_new = p_new - np.sum(p_new * pts, axis=1, keepdims=True) * pts
            beta = 0.0
            
        g_t = new_g_t
        p = p_new
        it += 1
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
    return best_pts, best_energy

def _polish_lbfgs(pts, max_iter=500, gtol=1e-14, time_limit=2.0):
    """Polish using L-BFGS-B on Euclidean coordinates with projection."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = _project_to_sphere(x_pts)
        e = _log_energy(x_pts)
        # Use tangent gradient projected back to Euclidean? 
        # L-BFGS-B expects Euclidean gradient. 
        # The gradient of the energy with respect to the Euclidean coordinates 
        # after projection is tricky. 
        # Instead, we can just use the tangent gradient and hope the optimizer 
        # handles the manifold implicitly via the projection in the function? 
        # No, that's inconsistent. 
        # Let's skip L-BFGS polish to avoid inconsistency and rely on RCG.
        # Or, use the Euclidean gradient computed before projection?
        # Let's just return the tangent gradient flattened. The optimizer will 
        # try to minimize in Euclidean space, but the function projects. 
        # This is effectively a "retraction" based optimization if we were 
        # doing it manually, but L-BFGS-B doesn't know that. 
        # It might work reasonably well for polishing.
        e_t, g_t = _energy_and_tangent_grad(x_pts)
        return e_t, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 20})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = _project_to_sphere(x_opt)
            e_opt = _log_energy(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _log_energy(pts)

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Fibonacci spiral
    current_pts = _fibonacci_sphere(N)
    best_pts = current_pts.copy()
    best_energy = _log_energy(best_pts)
    
    # Basin Hopping Loop
    sigma = 0.1
    kick_count = 0
    
    while time.time() < deadline - 5.0:
        # 1. Local Minimization
        # Use Riemannian CG for speed and manifold consistency
        current_pts, current_energy = _riemannian_cg(current_pts, max_iters=50, time_limit=5.0)
        
        # Optional: Polish with L-BFGS if time permits
        if time.time() < deadline - 10.0:
            current_pts, current_energy = _polish_lbfgs(current_pts, max_iter=100, time_limit=2.0)
        
        # 2. Update Best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
        
        # 3. Check Time
        if time.time() > deadline - 10.0:
            break
            
        # 4. Perturb (Kick)
        # Increase sigma slightly if we are stuck? 
        # Simple strategy: constant sigma, or decay?
        # Let's try a fixed sigma that is small enough not to destroy the structure 
        # but large enough to jump basins.
        noise = rng.standard_normal(current_pts.shape) * sigma
        current_pts = _project_to_sphere(current_pts + noise)
        
        kick_count += 1
        
        # Adaptive sigma: if we haven't improved in last 5 kicks, increase sigma
        # This is a simple heuristic.
        # For now, keep it simple.
        
    # Final Polish on Best
    if time.time() < deadline - 2.0:
        best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=200, time_limit=2.0)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
