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
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j log(dist[i,j])
    # Use math.fsum for consistency with evaluator
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu].tolist()
    energy = -math.fsum([math.log(d) for d in dist_upper])
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _sphere_energy_and_grad(x_flat):
    """
    Objective and gradient for scipy.optimize.minimize.
    x_flat is a 3*N array of coordinates.
    We normalize each point to the sphere inside this function.
    """
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    
    # Project gradient onto tangent space for the effective gradient
    # For f(x) = E(x/||x||), the gradient w.r.t x is (I - x x^T/||x||^2) * grad_E
    # Since pts is already normalized (||pts||=1), the projection is:
    # grad_T = grad - (grad . pts) * pts
    proj = np.einsum('ij,ij->i', grad, pts)
    grad_tangent = grad - proj[:, np.newaxis] * pts
    
    return energy, grad_tangent.flatten()

def _retract(pts, grad_tangent, step_size):
    """
    Retraction on the sphere: move along tangent direction and re-normalize.
    """
    new_pts = pts - step_size * grad_tangent
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_pts = new_pts / np.maximum(norms, 1e-15)
    return new_pts

def _lbfgs_sphere(x0, deadline, maxiter=5000):
    """
    Run L-BFGS-B on the sphere with time guard.
    Three-phase: coarse, medium, fine tolerances for deeper convergence.
    """
    start_time = time.time()
    
    best_x = x0.copy()
    best_f, _ = _sphere_energy_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = _sphere_energy_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    # Phase 1: coarse L-BFGS - get into the basin quickly
    remaining = deadline - time.time()
    maxiter_p1 = min(maxiter, int(remaining * 20))
    if maxiter_p1 < 150:
        maxiter_p1 = 150
    
    try:
        res = minimize(
            _sphere_energy_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter_p1,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxfun': 10000,
                'maxls': 30
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Phase 2: medium tolerance
    remaining = deadline - time.time()
    if remaining > 5.0:
        maxiter_p2 = min(3000, int(remaining * 25))
        if maxiter_p2 < 100:
            maxiter_p2 = 100
        
        try:
            res2 = minimize(
                _sphere_energy_and_grad,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter_p2,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 8000,
                    'maxls': 30
                },
                callback=callback
            )
            if res2.fun < best_f:
                best_f = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: fine tolerance for deep convergence
    remaining = deadline - time.time()
    if remaining > 3.0:
        maxiter_p3 = min(2000, int(remaining * 30))
        if maxiter_p3 < 50:
            maxiter_p3 = 50
        
        try:
            res3 = minimize(
                _sphere_energy_and_grad,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter_p3,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxfun': 5000,
                    'maxls': 20
                },
                callback=callback
            )
            if res3.fun < best_f:
                best_f = res3.fun
                best_x = res3.x.copy()
        except Exception:
            pass
    
    return best_x, best_f

def _polish(pts, deadline, max_iter=200):
    """
    High-precision polish using gradient descent with backtracking line search
    on the tangent space.
    """
    start_time = time.time()
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_tangent = g - proj[:, np.newaxis] * pts
    
    lr = 0.05
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Backtracking line search with Armijo condition
        alpha = lr
        best_alpha = None
        best_f_new = f
        best_new_pts = None
        best_new_g = None
        best_new_g_t = None
        
        for ls in range(30):
            new_pts = _retract(pts, g_tangent, alpha)
            new_f, new_g = _energy_and_grad(new_pts)
            new_proj = np.einsum('ij,ij->i', new_g, new_pts)
            new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
            
            if new_f < best_f_new:
                best_f_new = new_f
                best_alpha = alpha
                best_new_pts = new_pts
                best_new_g = new_g
                best_new_g_t = new_g_tangent
            
            # Armijo condition
            if new_f < f - 1e-4 * alpha * np.sum(g_tangent * g_tangent):
                break
            
            alpha *= 0.5
        
        if best_alpha is None:
            break
        
        pts = best_new_pts
        f = best_f_new
        g = best_new_g
        g_tangent = best_new_g_t
        lr = min(best_alpha * 1.1, 0.08)
    
    return pts, f

def _rcg_polish(pts, deadline, max_iter=100):
    """
    Riemannian Conjugate Gradient polish for deeper local optimization.
    Uses Fletcher-Reeves beta with line search on the sphere.
    """
    if time.time() > deadline - 3.0:
        return pts, _energy_and_grad(pts)[0]
    
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    d = g - proj[:, np.newaxis] * pts  # gradient in tangent space
    d = -d  # negative gradient direction
    
    # Ensure d is in tangent space
    proj_d = np.einsum('ij,ij->i', d, pts)
    d = d - proj_d[:, np.newaxis] * pts
    
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-14:
        return pts, f
    
    d_prev = None
    g_prev = None
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Line search: find step size alpha
        alpha = 0.1
        best_alpha = None
        best_f_new = f
        
        for ls in range(20):
            new_pts = _retract(pts, d, alpha)
            new_f, new_g = _energy_and_grad(new_pts)
            new_proj = np.einsum('ij,ij->i', new_g, new_pts)
            new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
            
            if new_f < best_f_new:
                best_f_new = new_f
                best_alpha = alpha
                best_new_pts = new_pts
                best_new_g = new_g
                best_new_g_t = new_g_tangent
            
            if new_f < f - 1e-4 * alpha * np.sum(d * d):
                break
            
            alpha *= 0.5
        
        if best_alpha is None:
            break
        
        # Update
        pts = best_new_pts
        g = best_new_g
        f = best_f_new
        g_tangent = best_new_g_t
        
        # Fletcher-Reeves beta
        if d_prev is not None:
            g_dot_g = np.sum(g_tangent * g_tangent)
            g_prev_dot_g_prev = np.sum(g_prev * g_prev)
            if g_prev_dot_g_prev > 1e-30:
                beta = g_dot_g / g_prev_dot_g_prev
            else:
                beta = 0.0
            d = -g_tangent + beta * d
        else:
            d = -g_tangent
        
        # Ensure d is in tangent space
        proj_d = np.einsum('ij,ij->i', d, pts)
        d = d - proj_d[:, np.newaxis] * pts
        
        d_prev = d.copy()
        g_prev = g_tangent.copy()
        
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-14:
            break
    
    return pts, f

def _ultra_polish(pts, deadline, max_iter=800):
    """
    Ultra-fine polish with adaptive step sizing and backtracking for final convergence.
    """
    start_time = time.time()
    f, g = _energy_and_grad(pts)
    proj = np.einsum('ij,ij->i', g, pts)
    g_tangent = g - proj[:, np.newaxis] * pts
    
    lr = 0.01
    for it in range(max_iter):
        if time.time() > deadline - 0.5:
            break
        
        # Check convergence
        g_norm = np.linalg.norm(g_tangent)
        if g_norm < 1e-15:
            break
        
        # Backtracking line search
        alpha = lr
        found = False
        for ls in range(30):
            new_pts = _retract(pts, g_tangent, alpha)
            new_f, new_g = _energy_and_grad(new_pts)
            new_proj = np.einsum('ij,ij->i', new_g, new_pts)
            new_g_tangent = new_g - new_proj[:, np.newaxis] * new_pts
            
            if new_f < f:
                pts = new_pts
                f = new_f
                g = new_g
                g_tangent = new_g_tangent
                found = True
                break
            alpha *= 0.5
        
        if found:
            # Increase learning rate if we took the full step
            if alpha == lr:
                lr = min(lr * 1.08, 0.03)
            else:
                lr = alpha * 1.15
        else:
            lr *= 0.4
            if lr < 1e-18:
                break
    
    return pts, f

def _lbfgs_sphere_custom(pts, max_iter=500, tol=1e-12, deadline=None, m=25):
    """
    Custom L-BFGS on the sphere using Riemannian gradient and retraction.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
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
            new_proj = np.einsum('ij,ij->i', new_g, candidate)
            new_g_t = new_g - new_proj[:, np.newaxis] * candidate
            
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
        proj = np.einsum('ij,ij->i', g, x)
        g_t = g - proj[:, np.newaxis] * x
    
    return x, f

def construct():
    start_time = time.time()
    deadline = start_time + 147.0

    # Generate multiple diverse initializations
    initializations = []

    # 1. Fibonacci spiral (golden angle)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_pts)

    # 2. Random points (seeded for reproducibility)
    for seed_offset in range(4):
        rng_local = np.random.default_rng(42 + seed_offset)
        random_pts = rng_local.standard_normal((N, 3))
        norms = np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts = random_pts / np.maximum(norms, 1e-15)
        initializations.append(random_pts)

    # 3. Perturbed Fibonacci
    rng_pert = np.random.default_rng(123)
    for pert_scale in [0.01, 0.05]:
        perturbed = fib_pts + rng_pert.standard_normal((N, 3)) * pert_scale
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.maximum(norms, 1e-15)
        initializations.append(perturbed)

    # 4. Shifted Fibonacci
    for angle_shift in [0.1, 0.2]:
        i2 = np.arange(N) + 0.5
        z2 = 1.0 - 2.0 * i2 / N
        phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 + angle_shift
        r2 = np.sqrt(1.0 - z2 * z2)
        shifted_pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
        initializations.append(shifted_pts)

    # Optimize each initialization and keep the best
    best_x = None
    best_f = float('inf')

    n_inits = len(initializations)
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        x0 = init_pts.flatten()
        
        # Quick check: how much time do we have?
        remaining = deadline - time.time()
        if remaining < 20.0:
            break
        
        # Allocate time: give more time to earlier (likely better) initializations
        # but ensure we can try all of them
        time_per_init = max(15.0, (deadline - time.time() - 20.0) / max(1, n_inits - idx))
        local_deadline = time.time() + min(time_per_init, remaining - 20.0)
        
        x_opt, f_opt = _lbfgs_sphere(x0, local_deadline, maxiter=3000)
        
        if f_opt < best_f:
            best_f = f_opt
            best_x = x_opt.copy()

    if best_x is None:
        # Fallback to Fibonacci
        best_x = fib_pts.flatten()
        best_f, _ = _sphere_energy_and_grad(best_x)

    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        pts = best_x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        
        # First pass: standard gradient polish
        pts, f = _polish(pts, deadline, max_iter=300)
        best_x = pts.flatten()
        best_f = f
        
        # Second pass: Riemannian conjugate gradient for deeper optimization
        if time.time() < deadline - 3.0:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _rcg_polish(pts, deadline, max_iter=150)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
        
        # Third pass: Custom L-BFGS for deeper convergence
        if time.time() < deadline - 3.0:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _lbfgs_sphere_custom(pts, max_iter=500, tol=1e-12, deadline=deadline - 1.0)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
        
        # Fourth pass: another round of gradient polish after RCG
        if time.time() < deadline - 1.5:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _polish(pts, deadline, max_iter=200)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
        
        # Fifth pass: ultra-fine polish with adaptive steps for maximum precision
        if time.time() < deadline - 0.5:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _ultra_polish(pts, deadline, max_iter=1000)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
        
        # Sixth pass: final ultra-fine polish if time remains
        if time.time() < deadline - 0.2:
            pts = best_x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-15)
            pts, f = _ultra_polish(pts, deadline, max_iter=500)
            if f < best_f:
                best_x = pts.flatten()
                best_f = f
    
    # Ensure points are on the sphere
    pts_opt = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    pts_opt = pts_opt / np.maximum(norms, 1e-15)
    
    return pts_opt
# EVOLVE-BLOCK-END
