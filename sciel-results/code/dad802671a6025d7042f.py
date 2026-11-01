"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _retract(pts, direction, step_size):
    """
    Retraction on the sphere: move along tangent direction and re-normalize.
    """
    new_pts = pts + step_size * direction
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _tangent_project(grad, pts):
    """Project gradient onto tangent space."""
    proj = np.einsum('ij,ij->i', grad, pts)
    return grad - proj[:, np.newaxis] * pts

def _riemannian_cg(pts, deadline, max_iter=2000, tol=1e-14, m=15):
    """
    Riemannian Conjugate Gradient with adaptive backtracking and 
    damped Fletcher-Reeves beta.
    """
    x = pts.copy()
    f, g = _energy_and_grad(x)
    g_t = _tangent_project(g, x)
    g_norm = np.linalg.norm(g_t)
    
    if g_norm < tol:
        return x, f
    
    # Initial search direction
    d = -g_t
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:
        return x, f
    d = d / d_norm * g_norm  # Scale direction
    
    best_x = x.copy()
    best_f = f
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        # Backtracking line search (Armijo)
        step_size = 1.0
        sufficient = 1e-12 * step_size * np.sum(g_t * d)
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        
        for _ in range(60):
            candidate = _retract(x, d, step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_g_t = _tangent_project(new_g, candidate)
            
            if new_f <= f - 1e-12 * step_size * np.sum(g_t * d):
                new_x = candidate
                break
            step_size *= 0.5
        
        if new_f < best_f:
            best_f = new_f
            best_x = new_x.copy()
        
        # Damped Fletcher-Reeves beta
        old_g_t_norm_sq = np.sum(g_t * g_t)
        new_g_t_norm_sq = np.sum(new_g_t * new_g_t)
        
        # Damped FR: beta = max(0, (new_g^T new_g) / (old_g^T old_g) * damping)
        if old_g_t_norm_sq > 1e-30:
            beta_fr = new_g_t_norm_sq / old_g_t_norm_sq
        else:
            beta_fr = 0.0
        
        # Damping factor to ensure beta is not too large
        damping = 0.9
        beta = damping * min(beta_fr, 1.0)
        
        # PRP+ as alternative if FR is too aggressive
        prp_num = np.sum(new_g_t * (new_g_t - g_t))
        if prp_num < 0:
            prp_num = 0.0
        if old_g_t_norm_sq > 1e-30:
            beta_prp = prp_num / old_g_t_norm_sq
        else:
            beta_prp = 0.0
        
        # Use the smaller of FR and PRP+ (hybrid)
        beta = min(beta, beta_prp)
        
        # Ensure non-negative
        beta = max(0.0, beta)
        
        # New search direction
        d = -new_g_t + beta * d
        
        # Ensure descent direction
        if np.sum(new_g_t * d) > 0:
            d = -new_g_t
        
        x = new_x
        f = new_f
        g = new_g
        g_t = new_g_t
    
    return best_x, best_f

def _lbfgs_sphere_custom(pts, max_iter=300, tol=1e-13, deadline=None, m=20):
    """
    Custom L-BFGS on the sphere using Riemannian gradient and retraction.
    """
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    g_t = _tangent_project(g, x)
    
    s_list = []
    y_list = []
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        if len(s_list) > 0:
            rhos = []
            for s, y in zip(s_list, y_list):
                sy = np.sum(s * y)
                rhos.append(1.0 / sy if abs(sy) > 1e-15 else 0.0)
            
            q = g_t.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                alpha_i = rho_i * np.sum(s_i * q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            gamma = 1.0
            r = -gamma * q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                beta_i = rho_i * np.sum(y_i * r)
                r = r + (alphas[i] - beta_i) * s_i
            
            direction = r
        else:
            direction = -g_t
        
        # Line search with backtracking
        step_size = 1.0
        new_x = x
        new_f = f
        new_g = g
        new_g_t = g_t
        for _ in range(50):
            candidate = _retract(x, direction, step_size)
            new_f, new_g = _energy_and_grad(candidate)
            new_g_t = _tangent_project(new_g, candidate)
            
            if new_f <= f + 1e-5 * step_size * np.sum(g_t * direction):
                new_x = candidate
                break
            step_size *= 0.5
        else:
            new_x = candidate
        
        s_k = new_x - x
        y_k = new_g_t - g_t
        sy = np.sum(s_k * y_k)
        if sy > 1e-12:
            s_list.append(s_k)
            y_list.append(y_k)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = new_x
        f = new_f
        g = new_g
        g_t = new_g_t
    
    return x, f

def _polish(pts, deadline, max_iter=1000, lr_init=0.03, lr_max=0.1):
    """
    High-precision polish using gradient descent with backtracking line search
    on the tangent space. Uses adaptive learning rate.
    """
    f, g = _energy_and_grad(pts)
    g_tangent = _tangent_project(g, pts)
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        new_pts = _retract(pts, -g_tangent, lr)
        new_f, new_g = _energy_and_grad(new_pts)
        new_g_tangent = _tangent_project(new_g, new_pts)
        
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
            g_tangent = new_g_tangent
            lr = min(lr * 1.1, lr_max)
        else:
            lr *= 0.5
            if lr < 1e-15:
                break
        
        if np.linalg.norm(g_tangent) < 1e-15:
            break
    
    return pts, f

def _ultra_polish(pts, deadline, max_iter=3000):
    """
    Ultra-precise polish with very small steps and tight Armijo.
    """
    f, g = _energy_and_grad(pts)
    g_t = _tangent_project(g, pts)
    
    g_norm = np.linalg.norm(g_t)
    if g_norm < 1e-15:
        return pts, f
    
    alpha = 0.005
    for it in range(max_iter):
        if time.time() > deadline - 0.5:
            break
        
        sufficient = 1e-16 * alpha * np.sum(g_t * (-g_t))
        cand = _retract(pts, -g_t, alpha)
        cand_f, cand_g = _energy_and_grad(cand)
        cand_g_t = _tangent_project(cand_g, cand)
        
        if cand_f <= f - sufficient:
            pts = cand
            f = cand_f
            g = cand_g
            g_t = cand_g_t
            if it % 20 == 0:
                alpha = min(alpha * 1.03, 0.02)
        else:
            alpha *= 0.5
            if alpha < 1e-20:
                break
        
        if np.linalg.norm(g_t) < 1e-16:
            break
    
    return pts, f

def _fibonacci_spiral(offset=0.0):
    """Generate Fibonacci spiral initial points with optional offset."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _golden_spiral(offset=0.0):
    """Generate golden ratio spiral initial points."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = 2.0 * np.pi * i / (1.0 + np.sqrt(5.0))
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate n random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _perturb(pts, magnitude=0.05, seed=None):
    """Add small random perturbation to escape local minima."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(pts.shape) * magnitude
    new_pts = pts + noise
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _deduplicate(pts_list, threshold=0.2):
    """Remove initializations that are too similar to already-kept ones."""
    kept = []
    for pts in pts_list:
        too_close = False
        for kp in kept:
            diff = pts[:, np.newaxis, :] - kp[np.newaxis, :, :]
            dmin = np.min(np.linalg.norm(diff, axis=2))
            if dmin < threshold:
                too_close = True
                break
        if not too_close:
            kept.append(pts)
    return kept

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_fibonacci_spiral())
    inits.append(_golden_spiral())
    inits.append(_fibonacci_spiral(offset=0.25))
    inits.append(_golden_spiral(offset=0.3))
    
    # Add several random starts with different seeds
    for seed in [42, 123, 456, 789, 101112, 202021, 303031]:
        inits.append(_random_sphere(N, seed=seed))
    
    # Deduplicate
    inits = _deduplicate(inits, threshold=0.15)
    
    best_pts = None
    best_f = np.inf
    
    # Optimize each initialization
    for idx, init in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
        
        # Allocate time proportionally
        time_per_init = min(remaining / max(1, len(inits) - idx), 25.0)
        init_deadline = time.time() + time_per_init
        
        # Primary: Riemannian CG
        pts_opt, f_opt = _riemannian_cg(init, init_deadline, max_iter=1500, tol=1e-14)
        
        # Secondary: Custom L-BFGS for additional refinement
        if time.time() < init_deadline - 3.0:
            pts_opt, f_opt = _lbfgs_sphere_custom(pts_opt, max_iter=200, tol=1e-14,
                                                   deadline=init_deadline - 2.0)
        
        # Polish if time permits
        if time.time() < init_deadline - 2.0:
            pts_opt, f_opt = _polish(pts_opt, init_deadline, max_iter=500)
        
        # Ultra-polish
        if time.time() < init_deadline - 0.5:
            pts_opt, f_opt = _ultra_polish(pts_opt, init_deadline, max_iter=1500)
        
        # Perturbation-based escape: try a few perturbations if we have time
        if time.time() < deadline - 5.0 and f_opt < best_f:
            for pseed in [7, 17, 27]:
                if time.time() > deadline - 3.0:
                    break
                perturbed = _perturb(pts_opt, magnitude=0.03, seed=pseed)
                p_deadline = min(init_deadline, deadline - 2.0)
                pts_p, f_p = _riemannian_cg(perturbed, p_deadline, max_iter=500, tol=1e-13)
                if time.time() < p_deadline - 1.0:
                    pts_p, f_p = _polish(pts_p, p_deadline, max_iter=200)
                if f_p < f_opt:
                    pts_opt = pts_p
                    f_opt = f_p
        
        if f_opt < best_f:
            best_f = f_opt
            best_pts = pts_opt.copy()
    
    # Final refinement on the best configuration
    if best_pts is not None and time.time() < deadline - 4.0:
        # Custom L-BFGS for additional refinement
        remaining = deadline - time.time()
        if remaining > 5.0:
            pts_ref, f_ref = _lbfgs_sphere_custom(best_pts, max_iter=150, tol=1e-15,
                                                   deadline=deadline - 2.0)
            if f_ref < best_f:
                best_pts = pts_ref
                best_f = f_ref
        
        # Riemannian CG final pass
        if time.time() < deadline - 2.0:
            pts_cg, f_cg = _riemannian_cg(best_pts, deadline - 1.0, max_iter=500, tol=1e-15)
            if f_cg < best_f:
                best_pts = pts_cg
                best_f = f_cg
        
        # Final ultra-polish
        if time.time() < deadline - 1.0:
            best_pts, best_f = _ultra_polish(best_pts, deadline, max_iter=2000)
    
    # Ensure points are on the sphere and valid
    if best_pts is None:
        best_pts = _fibonacci_spiral()
    
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)
    
    return best_pts
# EVOLVE-BLOCK-END
