"""Rewritten program: CG on sphere with improved line search, diverse inits, and L-BFGS-B manifold polish."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and tangent gradient on the sphere.
    pts: (N, 3) array of points assumed to be on the unit sphere.
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    
    energy = -0.25 * np.sum(log_dist_sq)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    term1 = inv_dist_sq @ pts
    term2 = pts * np.sum(inv_dist_sq, axis=1)[:, None]
    
    grad_raw = 0.5 * (term1 - term2)
    
    dot = np.sum(grad_raw * pts, axis=1)
    grad_tan = grad_raw - dot[:, None] * pts
    
    return energy, grad_tan

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_golden_spiral():
    """Golden ratio spiral initialization."""
    i = np.arange(N)
    # Use golden angle for more uniform distribution
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / N
    phi = golden_angle * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_rackard():
    """Rackard's construction: take 2D optimal points and map to sphere."""
    # Start with a good 2D configuration (hexagonal lattice subset)
    pts_2d = []
    # Use a simple grid-like approach in 2D then project
    # Actually, let's just use a perturbed Fibonacci
    pts = _initial_fibonacci()
    rng = np.random.default_rng(123)
    pts = pts + rng.normal(size=(N, 3)) * 0.05
    return _project_to_sphere(pts)

def _conjugate_gradient(pts, max_iter=2000, tol=1e-12, deadline=None):
    """
    Conjugate Gradient method on the sphere with two-phase Wolfe line search.
    """
    x = pts.copy()
    energy, grad = _energy_and_tangent_grad(x)
    p = -grad
    
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    
    grad_prev = grad.copy()
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Two-phase Wolfe line search
        c1 = 1e-4
        c2 = 0.9
        rho_shrink = 0.5
        rho_expand = 1.5
        max_alpha = 50.0
        min_alpha = 1e-10
        
        dir_deriv = np.sum(grad * p)
        
        if dir_deriv > 0:
            p = -grad
            dot_p_x = np.sum(p * x, axis=1)
            p = p - dot_p_x[:, None] * x
            dir_deriv = np.sum(grad * p)
        
        found_step = False
        alpha = 1.0
        
        # Phase 1: Explore - try to find a step that satisfies Armijo
        # Start with alpha=1.0 and expand if needed
        for phase1_iter in range(30):
            if alpha > max_alpha:
                alpha = max_alpha
                break
            if alpha < min_alpha:
                break
                
            x_new = x + alpha * p
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_tangent_grad(x_new)
            
            # Armijo condition
            if e_new <= energy + c1 * alpha * dir_deriv:
                # Phase 2: Check curvature
                new_dir_deriv = np.sum(g_new * p)
                if new_dir_deriv <= c2 * dir_deriv:
                    found_step = True
                    break
                # If curvature fails, try smaller step
                alpha *= rho_shrink
            else:
                # Try larger step (explore)
                alpha *= rho_expand
        
        if not found_step:
            # Fallback: try a small step
            alpha = 0.01
            for _ in range(20):
                x_new = x + alpha * p
                x_new = _project_to_sphere(x_new)
                e_new, g_new = _energy_and_tangent_grad(x_new)
                if e_new <= energy + c1 * alpha * dir_deriv:
                    found_step = True
                    break
                alpha *= 0.5
            
            if not found_step:
                break
        
        x = x_new
        energy = e_new
        grad = g_new
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Polak-Ribiere with restart
        grad_dot = np.sum(grad * (grad - grad_prev))
        grad_prev_dot = np.sum(grad_prev * grad_prev)
        
        if grad_prev_dot < 1e-16:
            beta = 0.0
        else:
            beta = max(0.0, grad_dot / grad_prev_dot)
            
        p_new = -grad + beta * p
        
        dot_p_new_x = np.sum(p_new * x, axis=1)
        p_new = p_new - dot_p_new_x[:, None] * x
        
        if np.sum(grad * p_new) > 0:
            beta = 0.0
            p_new = -grad
            dot_p_new_x = np.sum(p_new * x, axis=1)
            p_new = p_new - dot_p_new_x[:, None] * x
            
        p = p_new
        grad_prev = grad.copy()
        
    return x

def _lbfgs_manifold_polish(pts, max_iter=500, deadline=None):
    """
    L-BFGS-B polish using retraction-based manifold optimization.
    We optimize in R^{3N} where f(x) = E(project(x)), and return
    the correct gradient for this composite function.
    """
    def objective_and_grad(x_flat):
        if deadline and time.time() > deadline:
            return 0, np.zeros_like(x_flat)
        x = x_flat.reshape(N, 3)
        x_proj = _project_to_sphere(x)
        energy, grad_tan = _energy_and_tangent_grad(x_proj)
        # The gradient of f(x) = E(x/||x||) w.r.t. x is grad_tan (as derived)
        return energy, grad_tan.flatten()
    
    x0 = pts.flatten()
    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 10000}
        )
        x_opt = res.x.reshape(N, 3)
        x_opt = _project_to_sphere(x_opt)
        return x_opt
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    inits = []
    inits.append(_initial_fibonacci())
    inits.append(_initial_golden_spiral())
    inits.append(_initial_rackard())
    
    rng = np.random.default_rng(42)
    
    # Random initializations
    for i in range(8):
        pts = rng.normal(size=(N, 3))
        inits.append(_project_to_sphere(pts))
    
    # Perturbed Fibonacci
    for i in range(8):
        noise_scale = 0.05 + 0.02 * i
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * noise_scale
        inits.append(_project_to_sphere(pts))
    
    # Perturbed golden spiral
    for i in range(4):
        noise_scale = 0.05 + 0.03 * i
        pts = _initial_golden_spiral() + rng.normal(size=(N, 3)) * noise_scale
        inits.append(_project_to_sphere(pts))
    
    # Block-structured perturbations
    for i in range(4):
        pts = _initial_fibonacci().copy()
        # Perturb random subsets more
        for _ in range(3):
            idx = rng.choice(N, size=N//4, replace=False)
            pts[idx] += rng.normal(size=(len(idx), 3)) * 0.1
        inits.append(_project_to_sphere(pts))
    
    best_pts = inits[0].copy()
    best_energy = _energy_and_tangent_grad(best_pts)[0]
    
    # Phase 1: CG optimization on all initializations
    # Allocate ~60% of time to exploration
    explore_deadline = start_time + 0.6 * (deadline - start_time)
    
    for i, init_pts in enumerate(inits):
        if time.time() > explore_deadline - 2:
            break
            
        time_left = explore_deadline - time.time()
        if time_left < 3:
            break
            
        run_deadline = time.time() + min(4.0, time_left - 1.0)
        if run_deadline <= time.time():
            continue
            
        optimized_pts = _conjugate_gradient(init_pts, max_iter=1000, tol=1e-12, deadline=run_deadline)
        
        e_opt, _ = _energy_and_tangent_grad(optimized_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = optimized_pts.copy()
    
    # Phase 2: Polish the best solution with more CG iterations
    if time.time() < deadline - 20:
        polish_deadline = time.time() + min(30.0, deadline - time.time() - 15)
        polished_pts = _conjugate_gradient(best_pts, max_iter=5000, tol=1e-14, deadline=polish_deadline)
        e_polished, _ = _energy_and_tangent_grad(polished_pts)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = polished_pts.copy()
    
    # Phase 3: L-BFGS-B manifold polish for high precision
    if time.time() < deadline - 10:
        lbfgs_deadline = time.time() + min(20.0, deadline - time.time() - 5)
        lbfgs_pts = _lbfgs_manifold_polish(best_pts, max_iter=300, deadline=lbfgs_deadline)
        e_lbfgs, _ = _energy_and_tangent_grad(lbfgs_pts)
        if e_lbfgs < best_energy:
            best_energy = e_lbfgs
            best_pts = lbfgs_pts.copy()
    
    # Phase 4: Final tight CG polish if time remains
    if time.time() < deadline - 5:
        final_deadline = deadline - 2
        final_pts = _conjugate_gradient(best_pts, max_iter=3000, tol=1e-14, deadline=final_deadline)
        e_final, _ = _energy_and_tangent_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
    
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9 or not np.all(np.isfinite(best_pts)):
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
