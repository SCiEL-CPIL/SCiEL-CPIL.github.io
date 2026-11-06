"""Logarithmic energy minimisation for N points on the unit sphere."""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150
# Deadline guard: leave a safety margin (10 s) under the 150 s limit
_DEADLINE = time.time() + 140.0

def _energy(pts):
    """Vectorized sum_{i<j} log(1 / |x_i - x_j|) using math.fsum."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = d_sq[d_sq > 1e-30]
    if len(d_sq) == 0:
        return 1e18
    terms = -0.5 * np.log(d_sq)
    return math.fsum(terms)

def _grad(pts):
    """
    Vectorized gradient of E = sum_{i<j} log(1/|x_i - x_j|) = -sum log|d_ij|.
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    Tangent-projected.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)
    # grad_i = sum_j (x_i - x_j) / |x_i - x_j|^2
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)
    # Project to tangent space
    radial_comp = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial_comp * pts
    return grad

def _hessian_tangent(pts):
    """
    Approximate Hessian of E on the tangent space.
    For E = -sum_{i<j} log|d_ij|, the Hessian is complex.
    We use a finite-difference-free approach: the Hessian of -log|d| is
    (I/d^2 - 3*d*d^T/d^4) for each pair.
    For manifold optimization, we project.
    This is expensive (O(N^2 * 3x3) per point), so we only use it for final polish.
    Returns a sparse-like representation or we do CG with matrix-vector products.
    """
    pass  # We'll implement Hessian-vector product instead for efficiency

def _retract(pts):
    """Project points back to the sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _hess_vec(pts, v):
    """
    Hessian-vector product for the log energy on the tangent space.
    E = -sum_{i<j} log|d_ij|
    Hessian of -log|d| with respect to ambient coords:
    For pair (i,j): d = x_i - x_j, r = |d|
    d2(-log r)/dx_i dx_j = -(I/r - d*d^T/r^3)  ... actually let me derive properly.
    
    f = -log|d| where d = x_i - x_j
    df/dx_i = -d/|d|^2
    d2f/dx_i^2 = I/|d|^2 - d*d^T/|d|^4
    d2f/dx_i dx_j = -(I/|d|^2 - d*d^T/|d|^4)
    
    So for the full Hessian acting on a vector v (ambient), the contribution from pair (i,j) to the i-th block is:
    (I/r^2 - d*d^T/r^4) * v_i - (I/r^2 - d*d^T/r^4) * v_j
    and to the j-th block is the negative.
    
    We then project to tangent space.
    """
    m = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    r2 = dist_sq
    # A_ij = I/r^2 - d*d^T/r^4  (3x3 matrix for each pair)
    # We compute the Hessian-vector product without forming A explicitly.
    # For pair (i,j): contribution to result_i = A_ij @ (v_i - v_j)
    # where A_ij @ w = w/r^2 - (d . w) * d / r^4
    
    # v is shape (N, 3) - tangent vector
    result = np.zeros_like(v)
    
    # Only need upper triangle pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    d = diff[i_idx, j_idx]  # shape (M, 3)
    r2_pairs = r2[i_idx, j_idx]  # shape (M,)
    r2_pairs = np.maximum(r2_pairs, 1e-30)
    
    v_diff = v[i_idx] - v[j_idx]  # shape (M, 3)
    # A @ v_diff = v_diff/r^2 - (d . v_diff) * d / r^4
    d_dot_vd = np.sum(d * v_diff, axis=1, keepdims=True)  # shape (M, 1)
    A_vd = v_diff / r2_pairs[:, None] - d_dot_vd * d / (r2_pairs[:, None] ** 2)
    
    # Accumulate: result[i] += A_vd, result[j] -= A_vd
    np.add.at(result, i_idx, A_vd)
    np.add.at(result, j_idx, -A_vd)
    
    # Project to tangent space
    radial_comp = np.sum(result * pts, axis=1, keepdims=True)
    result = result - radial_comp * pts
    
    return result

def _lbfgs_optimize(pts, max_time=30.0, tol=1e-10):
    """Optimize using L-BFGS-B on the sphere via tangent-space parameterization."""
    t_start = time.time()
    pts = pts.copy()
    E = _energy(pts)
    best_pts = pts.copy()
    best_E = E
    
    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        x = _retract(x)
        return _energy(x)
    
    def gradient(x_flat):
        x = x_flat.reshape(N, 3)
        x = _retract(x)
        g = _grad(x)
        return g.flatten()
    
    x0 = pts.flatten()
    
    # Run L-BFGS-B with time-limited iterations
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'ftol': tol,
            'gtol': tol,
            'maxfun': 5000,
            'maxls': 50,
            'maxcor': 20
        }
    )
    
    if result.fun < best_E:
        x_opt = _retract(result.x.reshape(N, 3))
        E_opt = _energy(x_opt)
        if E_opt < best_E:
            best_pts = x_opt
            best_E = E_opt
    
    return best_pts, best_E

def _polish(pts, max_time=15.0):
    """Final high-precision polish: L-BFGS-B then Newton on tangent space."""
    t_start = time.time()
    pts = pts.copy()
    E = _energy(pts)
    best_pts = pts.copy()
    best_E = E
    
    # Phase 1: L-BFGS-B with tight tolerances
    def objective(x_flat):
        x = x_flat.reshape(N, 3)
        x = _retract(x)
        return _energy(x)
    
    def gradient(x_flat):
        x = x_flat.reshape(N, 3)
        x = _retract(x)
        g = _grad(x)
        return g.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={
            'maxiter': 10000,
            'ftol': 1e-14,
            'gtol': 1e-14,
            'maxfun': 10000,
            'maxls': 50,
            'maxcor': 30
        }
    )
    
    if result.fun < best_E:
        x_opt = _retract(result.x.reshape(N, 3))
        E_opt = _energy(x_opt)
        if E_opt < best_E:
            best_pts = x_opt
            best_E = E_opt
    
    # Phase 2: Newton's method on the tangent space
    if time.time() - t_start < max_time - 2.0:
        pts_new = best_pts.copy()
        E_new = best_E
        
        # Build orthonormal tangent basis per point
        # For each point x_i, find two tangent vectors t1_i, t2_i
        # We'll work in the ambient space with projection.
        
        for newton_iter in range(50):
            if time.time() - t_start > max_time - 1.0:
                break
            
            g = _grad(pts_new)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-14:
                break
            
            # Hessian-vector product via CG to solve H * delta = -g
            # Use conjugate gradient with Hessian-vector product
            # H is positive definite (we're at a local min)
            
            def hvp(v):
                return _hess_vec(pts_new, v)
            
            # CG to solve H * delta = -g
            # Initialize
            r = -g.copy()
            p_vec = r.copy()
            rs = np.sum(r * r)
            
            delta = np.zeros_like(g)
            
            for cg_iter in range(50):
                if time.time() - t_start > max_time - 1.0:
                    break
                Ap = hvp(p_vec)
                pAp = np.sum(p_vec * Ap)
                if abs(pAp) < 1e-30:
                    break
                alpha = rs / pAp
                delta += alpha * p_vec
                r -= alpha * Ap
                new_rs = np.sum(r * r)
                if np.sqrt(new_rs) < 1e-14:
                    break
                p_vec = r + (new_rs / rs) * p_vec
                rs = new_rs
            
            # Damped Newton step
            # Try full step first, then bisect if energy increases
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
            improved = False
            for step in step_sizes:
                candidate = pts_new + step * delta
                candidate = _retract(candidate)
                E_cand = _energy(candidate)
                if E_cand < E_new:
                    pts_new = candidate
                    E_new = E_cand
                    improved = True
                    break
            
            if not improved:
                break
            
            if E_new < best_E:
                best_pts = pts_new.copy()
                best_E = E_new
    
    return best_pts, best_E

def construct():
    t0 = time.time()
    
    # 1. Start with Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    best_pts, best_E = _lbfgs_optimize(pts0, max_time=20.0)
    
    # 2. Multiple diverse restarts
    seeds = [42, 123, 999, 2023, 777, 555, 314, 271, 161, 83]
    for seed in seeds:
        if time.time() > _DEADLINE - 25:
            break
        p = _random_sphere(N, seed)
        p2, E2 = _lbfgs_optimize(p, max_time=12.0)
        if E2 < best_E:
            best_pts, best_E = p2, E2
    
    # 3. Jittered Fibonacci for more diversity
    if time.time() < _DEADLINE - 20:
        rng = np.random.default_rng(12345)
        for _ in range(3):
            if time.time() > _DEADLINE - 18:
                break
            p = _fibonacci_sphere(N)
            p += rng.normal(0, 0.1, (N, 3))
            p = _retract(p)
            p2, E2 = _lbfgs_optimize(p, max_time=12.0)
            if E2 < best_E:
                best_pts, best_E = p2, E2
    
    # 4. Final high-precision polish with Newton
    if time.time() < _DEADLINE - 5:
        polished_pts, polished_E = _polish(best_pts, max_time=15.0)
        if polished_E < best_E:
            best_pts, best_E = polished_pts, polished_E

    # Final normalization
    best_pts = _retract(best_pts)
    
    return best_pts
